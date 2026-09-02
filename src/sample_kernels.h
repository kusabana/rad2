#pragma once

#include "geometry.h"
#include "lights.h"
#include "luxels.h"
#include "math.h"

#include <embree4/rtcore.h>

#include <cstdint>

inline float halton_element( uint32_t index, uint32_t base )
{
  float f = 1.0f;
  float r = 0.0f;

  while ( index > 0 )
  {
    f /= float( base );
    r += f * float( index % base );
    index /= base;
  }

  return r;
}

// mathlib/halton.cpp:13
inline vec3 halton_sphere( uint32_t k )
{
  const float z = 2.0f * halton_element( k, 2 ) - 1.0f;
  const float theta = 2.0f * glm::pi<float>() * halton_element( k, 3 );
  const float sin_sq = 1.0f - z * z;
  const float sin_p = glm::sqrt( sin_sq > 0.0f ? sin_sq : 0.0f );

  return vec3( glm::cos( theta ) * sin_p, glm::sin( theta ) * sin_p, z );
}

// utils/vrad/lightmap.cpp:1715
inline vec3 sun_jitter_direction( const vec3& to_sun, float extent_sin, uint32_t r )
{
  if ( r == 0 || extent_sin <= 0.0f )
    return to_sun;

  return safe_normalize( to_sun + halton_sphere( r ) * extent_sin );
}

struct light_eval
{
  vec3 contribution = vec3( 0.0f );
  vec3 to_light = vec3( 0.0f );
  float distance = 0.0f;
};

// point and spot lights
inline light_eval evaluate_standard_light(
    const light& source, const vec3& position, const vec3& normal )
{
  light_eval result;

  const vec3 delta = source.origin - position;
  const float dist2 = glm::dot( delta, delta );
  float dist = glm::sqrt( dist2 );
  if ( dist < length_epsilon )
    return result;

  const vec3 to_light = delta / dist;
  result.to_light = to_light;
  result.distance = dist;

  float n_dot_l = glm::dot( to_light, normal );
  if ( n_dot_l <= 0.0f )
    return result;

  const bool has_hard_falloff = source.end_fade > source.start_fade;
  if ( has_hard_falloff && dist > source.end_fade )
    return result;

  dist = glm::max( dist, 1.0f );
  const float falloff_eval_dist = glm::min( dist, source.cap_dist );

  float falloff = 1.0f / ( source.constant_attn + source.linear_attn * falloff_eval_dist +
                             source.quadratic_attn * falloff_eval_dist * falloff_eval_dist );

  if ( source.type == emit_type::spotlight )
  {
    const float dot2 = -glm::dot( to_light, source.normal );
    if ( dot2 <= source.stopdot2 )
      return result;

    falloff *= dot2;

    if ( dot2 <= source.stopdot )
    {
      float fringe = ( dot2 - source.stopdot2 ) / ( source.stopdot - source.stopdot2 );
      fringe = glm::clamp( fringe, 0.0f, 1.0f );

      if ( source.exponent != 0.0f && source.exponent != 1.0f )
        fringe = glm::pow( fringe, source.exponent );

      falloff *= fringe;
    }
  }

  if ( has_hard_falloff )
  {
    float t = ( dist - source.start_fade ) / ( source.end_fade - source.start_fade );
    t = 1.0f - glm::clamp( t, 0.0f, 1.0f );
    falloff *= t * t * t * ( t * ( t * 6.0f - 15.0f ) + 10.0f );
  }

  result.contribution = source.intensity * ( falloff * n_dot_l );
  return result;
}

namespace geom
{
enum : uint32_t
{
  occluder_id = 0, // brush sides and displacements
  sky_id = 1,
  face_lm_id = 2, // lightmapped face triangles

  occluder_mask = 0x1,
  sky_mask = 0x2,
  face_lm_mask = 0x10,

  mask_world = occluder_mask | sky_mask, // shadow, sun and sky rays
  mask_gather = occluder_mask | sky_mask | face_lm_mask,
};
}

constexpr float ray_t_min = 0.01f;

struct trace_ray
{
  vec3 origin;
  vec3 direction;
  float t_max;
  uint32_t mask;
};

constexpr RTCFeatureFlags ray_features =
    RTCFeatureFlags( RTC_FEATURE_FLAG_TRIANGLE | RTC_FEATURE_FLAG_32_BIT_RAY_MASK );

inline RTCRay embree_ray( const trace_ray& ray )
{
  RTCRay result;
  result.org_x = ray.origin.x;
  result.org_y = ray.origin.y;
  result.org_z = ray.origin.z;
  result.tnear = ray_t_min;
  result.dir_x = ray.direction.x;
  result.dir_y = ray.direction.y;
  result.dir_z = ray.direction.z;
  result.time = 0.0f;
  result.tfar = ray.t_max;
  result.mask = ray.mask;
  result.id = 0;
  result.flags = 0;
  return result;
}

struct ray_tracer
{
  RTCTraversable traversable;

  uint32_t closest_geometry( const trace_ray& ray ) const
  {
    RTCRayHit rayhit;
    rayhit.ray = embree_ray( ray );
    rayhit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    rayhit.hit.primID = RTC_INVALID_GEOMETRY_ID;

    RTCIntersectArguments args;
    rtcInitIntersectArguments( &args );
    args.feature_mask = ray_features;

    rtcTraversableIntersect1( traversable, &rayhit, &args );
    return rayhit.hit.geomID;
  }

  bool occluded( const trace_ray& ray ) const
  {
    RTCRay shadow = embree_ray( ray );

    RTCOccludedArguments args;
    rtcInitOccludedArguments( &args );
    args.feature_mask = ray_features;

    rtcTraversableOccluded1( traversable, &shadow, &args );
    return shadow.tfar < 0.0f;
  }
};

// public/worldsize.h:32
// constexpr float ray_t_max = coord_extent * const_sqrt( 3.0f );
constexpr float ray_t_max = coord_extent * 1.732050807569f;

struct sample_scene
{
  const luxel* luxels = nullptr;
  const face_info* faces = nullptr;
  const edge_plane* edge_planes = nullptr;
  const light* lights = nullptr;
  uint32_t light_count = 0;
  sky_light sky = {};
  uint32_t luxel_count = 0;
};

// per launch parameters
struct kernel_params
{
  uint32_t position_samples = 0;
  uint32_t sun_samples = 0;
  uint32_t sky_samples = 0;
};

constexpr int lift_precision_bits = 10;
constexpr float offset_scale_reference =
    float( 1 << ( float_mantissa_bits - lift_precision_bits ) );

inline float offset_scale( const vec3& position )
{
  return glm::max( 1.0f, max_component( glm::abs( position ) ) / offset_scale_reference );
}

// contributions below this skip their shadow ray
constexpr float light_cutoff = 0.001f;

// utils/vrad/lightmap.cpp:1782
constexpr float sky_cos_cutoff = equal_epsilon;

// utils/vrad/lightmap.cpp:2694
inline bool supersample_position( const sample_scene& scene, const luxel& sample,
    const face_info& face, uint32_t sample_index, uint32_t sample_count, vec3& position )
{
  const uint32_t width_shift = sample_count <= 4 ? 1 : 2;
  const uint32_t width = 1u << width_shift;
  const float cscale = 1.0f / float( width );
  const float csshift = -( float( width - 1 ) * cscale ) / 2.0f;
  const vec2 candidate =
      sample.luxel_position + vec2( csshift + float( sample_index & ( width - 1 ) ) * cscale,
                                  csshift + float( sample_index >> width_shift ) * cscale );

  // the cell clipped to the face
  const uint32_t local = uint32_t( &sample - scene.luxels ) - face.first_luxel;
  const float cell_s = float( local % uint32_t( face.width ) );
  const float cell_t = float( local / uint32_t( face.width ) );
  if ( candidate.x < cell_s || candidate.x > cell_s + 1.0f || candidate.y < cell_t ||
       candidate.y > cell_t + 1.0f )
    return false;

  for ( uint32_t e = 0; e < face.edge_count; ++e )
  {
    const edge_plane& edge = scene.edge_planes[face.edge_first + e];
    if ( glm::dot( edge.inward, candidate ) - edge.dist < 0.0f )
      return false;
  }

  const vec2 delta = candidate - sample.luxel_position;
  const vec3 s_axis = glm::cross( vec3( face.world_to_luxel_t ), face.plane_normal );
  const vec3 t_axis = glm::cross( face.plane_normal, vec3( face.world_to_luxel_s ) );
  const float det = -glm::dot( face.plane_normal,
      glm::cross( vec3( face.world_to_luxel_t ), vec3( face.world_to_luxel_s ) ) );

  if ( glm::abs( det ) < basis_epsilon )
    return false;

  position = sample.position + ( s_axis * delta.x + t_axis * delta.y ) / det;
  return true;
}

// utils/vrad/trace.cpp:389
inline bool skybox_clear( const sample_scene& scene, const ray_tracer& tracer, int32_t area,
    const vec3& origin, const vec3& direction )
{
  if ( scene.sky.cam_area < 0 || area == scene.sky.cam_area )
    return true;

  const vec3 sky_origin = scene.sky.cam_origin + origin * scene.sky.world_to_sky;
  const uint32_t hit =
      tracer.closest_geometry( { sky_origin, direction, ray_t_max, geom::mask_world } );

  return hit == RTC_INVALID_GEOMETRY_ID || hit == geom::sky_id;
}

struct direct_sample
{
  vec3 operator()( const sample_scene& scene, const ray_tracer& tracer, uint32_t luxel_index,
      const vec3& position, const kernel_params& params ) const
  {
    const luxel& sample = scene.luxels[luxel_index];
    const face_info& face = scene.faces[sample.face_index];
    const float scale = offset_scale( sample.position );
    const vec3 shadow_origin = position + face.plane_normal * ( sample_lift * scale );
    vec3 sum( 0.0f );

    for ( uint32_t li = 0; li < scene.light_count; ++li )
    {
      const light_eval eval =
          evaluate_standard_light( scene.lights[li], shadow_origin, sample.normal );
      if ( max_component( eval.contribution ) < light_cutoff )
        continue;

      if ( eval.distance > ray_t_min &&
           tracer.occluded( { shadow_origin, eval.to_light, eval.distance, geom::mask_world } ) )
      {
        continue;
      }

      sum += eval.contribution;
    }

    if ( max_component( scene.sky.sun_intensity ) > 0.0f )
    {
      const vec3 to_sun = -scene.sky.sun_normal;
      const float n_dot_l = glm::dot( sample.normal, to_sun );
      if ( n_dot_l > 0.0f )
      {
        const vec3 sky_origin = position + face.plane_normal * ( sample_lift * scale );
        const uint32_t rays = scene.sky.sun_extent_sin > 0.0f ? params.sun_samples : 1;
        uint32_t visible = 0;

        for ( uint32_t r = 0; r < rays; ++r )
        {
          const vec3 direction = sun_jitter_direction( to_sun, scene.sky.sun_extent_sin, r );
          if ( tracer.closest_geometry( { sky_origin, direction, ray_t_max, geom::mask_world } ) ==
                   geom::sky_id &&
               skybox_clear( scene, tracer, sample.area, sky_origin, direction ) )
          {
            ++visible;
          }
        }

        if ( visible > 0 )
          sum += scene.sky.sun_intensity * ( n_dot_l * float( visible ) / float( rays ) );
      }
    }

    return sum;
  }
};

// utils/vrad/lightmap.cpp:1746
struct sky_sample
{
  vec3 operator()( const sample_scene& scene, const ray_tracer& tracer, uint32_t luxel_index,
      const vec3& position, const kernel_params& params ) const
  {
    const luxel& sample = scene.luxels[luxel_index];
    const face_info& face = scene.faces[sample.face_index];
    const vec3 origin =
        position + face.plane_normal * ( sample_lift * offset_scale( sample.position ) );
    float sum_cos = 0.0f;
    float sum_visible = 0.0f;

    // utils/vrad/lightmap.cpp:1766
    for ( uint32_t r = 0; r < params.sky_samples; ++r )
    {
      const vec3 direction = -halton_sphere( r + 1 );
      const float cos_n = glm::dot( sample.normal, direction );
      if ( cos_n <= sky_cos_cutoff )
        continue;

      sum_cos += cos_n;

      if ( tracer.closest_geometry( { origin, direction, ray_t_max, geom::mask_world } ) ==
               geom::sky_id &&
           skybox_clear( scene, tracer, sample.area, origin, direction ) )
      {
        sum_visible += cos_n;
      }
    }

    if ( sum_cos <= 0.0f )
      return vec3( 0.0f );

    return scene.sky.ambient_intensity * ( sum_visible / sum_cos );
  }
};

template <typename evaluate_t>
inline void sample_kernel( const sample_scene& scene, const ray_tracer& tracer,
    uint32_t luxel_index, const kernel_params& params, vec3* out, evaluate_t evaluate )
{
  const luxel& sample = scene.luxels[luxel_index];
  out[luxel_index] = sample.world_area > 0.0f
                         ? evaluate( scene, tracer, luxel_index, sample.position, params )
                         : vec3( 0.0f );
}

// utils/vrad/lightmap.cpp:2943
template <typename evaluate_t>
inline void resample_position_kernel( const sample_scene& scene, const ray_tracer& tracer,
    const uint32_t* flagged, uint32_t f, uint32_t s, const kernel_params& params, vec3* slots,
    uint8_t* used, evaluate_t evaluate )
{
  const uint32_t luxel_index = flagged[f];
  const luxel& sample = scene.luxels[luxel_index];
  const uint32_t slot = f * params.position_samples + s;
  vec3 position;

  used[slot] = supersample_position(
      scene, sample, scene.faces[sample.face_index], s, params.position_samples, position );

  if ( used[slot] )
    slots[slot] = evaluate( scene, tracer, luxel_index, position, params );
}

template <typename evaluate_t>
inline void resample_average_kernel( const sample_scene& scene, const ray_tracer& tracer,
    const uint32_t* flagged, uint32_t f, const kernel_params& params, const vec3* slots,
    const uint8_t* used, vec3* results, evaluate_t evaluate )
{
  vec3 sum( 0.0f );
  uint32_t count = 0;

  for ( uint32_t s = 0; s < params.position_samples; ++s )
  {
    const uint32_t slot = f * params.position_samples + s;
    if ( used[slot] )
    {
      sum += slots[slot];
      ++count;
    }
  }

  if ( count == 0 )
  {
    sum = evaluate( scene, tracer, flagged[f], scene.luxels[flagged[f]].position, params );
    count = 1;
  }

  results[f] = sum / float( count );
}
