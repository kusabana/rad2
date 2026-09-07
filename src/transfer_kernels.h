#pragma once

#include "math.h"
#include "patches.h"
#include "sample_kernels.h"

#include <array>

// utils/vrad/vismat.cpp:34
constexpr float transfer_plane_epsilon = 0.01f;
// utils/vrad/vrad.h:64
constexpr float transfer_epsilon = 1.0e-7f;
// utils/vrad/vrad.h:184
constexpr uint32_t max_transfers = 4 * 65536;

constexpr uint32_t transfer_stack_depth = 128;

// utils/vrad/vismat.cpp:178
constexpr float descend_distance_ratio = 4.0f;

// utils/vrad/vrad.cpp:1157
constexpr float five_times_threshold = glm::pi<float>() * ( 1.0f / ( 5.0f * 5.0f ) );

constexpr float contour_sine_tolerance = 0.001f;

struct transfer_scene
{
  const patch_node* nodes = nullptr;
  const vec3* winding = nullptr;
  const uint32_t* cluster_first = nullptr;
  const uint32_t* cluster_faces = nullptr;
  const vec4* root_plane = nullptr;
  const int32_t* face_root = nullptr;
};

inline float contour_form_factor(
    const vec3* winding, uint32_t count, const vec3& point, const vec3& normal )
{
  float sum = 0.0f;

  for ( uint32_t i = 0; i < count; ++i )
  {
    const uint32_t next = i + 1 < count ? i + 1 : 0;
    const vec3 v1 = safe_normalize( winding[i] - point );
    const vec3 v2 = safe_normalize( winding[next] - point );
    vec3 gamma = glm::cross( v1, v2 );
    float sin_alpha = glm::length( gamma );
    if ( sin_alpha > 1.0f + contour_sine_tolerance )
      return 0.0f;

    sin_alpha = glm::min( sin_alpha, 1.0f );
    if ( sin_alpha > 0.0f )
      gamma *= glm::asin( sin_alpha ) / sin_alpha;

    sum += glm::dot( gamma, normal );
  }

  return sum * 0.5f;
}

inline float unnormalised_transfer(
    const transfer_scene& scene, const patch_node& receiver, const patch_node& shooter )
{
  if ( shooter.sky || shooter.area <= 0.0f )
    return 0.0f;

  vec3 delta = shooter.origin - receiver.origin;
  const float length_sq = glm::dot( delta, delta );
  const float length = glm::sqrt( length_sq );
  if ( length <= 0.0f )
    return 0.0f;

  delta /= length;

  const float scale =
      -glm::dot( delta, shooter.normal ) * glm::dot( delta, receiver.normal ) / length_sq;

  if ( scale <= 0.0f )
    return 0.0f;

  const float threshold = five_times_threshold * length_sq;
  float trans;

  if ( threshold < shooter.area )
  {
    const float contour = contour_form_factor( &scene.winding[shooter.winding_first],
        shooter.winding_count, receiver.origin, receiver.normal );

    if ( contour <= 0.0f )
      return 0.0f;

    trans = contour;
  }
  else
  {
    trans = shooter.area * scale;
  }

  if ( trans <= transfer_epsilon )
    return 0.0f;

  return trans;
}

inline uint32_t collect_transfer_candidates(
    const transfer_scene& scene, uint32_t receiver_index, uint32_t* out_node, float* out_trans )
{
  const patch_node& receiver = scene.nodes[receiver_index];
  const float receiver_dist = glm::dot( receiver.origin, receiver.normal );
  uint32_t count = 0;
  std::array<uint32_t, transfer_stack_depth> stack;

  for ( uint32_t k = scene.cluster_first[receiver.cluster];
      k < scene.cluster_first[receiver.cluster + 1]; ++k )
  {
    const uint32_t face = scene.cluster_faces[k];
    if ( face == receiver.face )
      continue;

    const vec4 plane = scene.root_plane[face];
    if ( glm::dot( receiver.origin, vec3( plane ) ) <= plane.w + transfer_plane_epsilon )
      continue;

    uint32_t depth = 0;
    stack[depth++] = uint32_t( scene.face_root[face] );

    while ( depth > 0 )
    {
      const uint32_t index = stack[--depth];
      const patch_node& node = scene.nodes[index];
      if ( !node.leaf() )
      {
        const vec3 delta = receiver.origin - node.origin;
        if ( glm::dot( delta, delta ) <
                 node.area * ( descend_distance_ratio * descend_distance_ratio ) &&
             depth + 2 <= transfer_stack_depth )
        {
          stack[depth++] = uint32_t( node.child[0] );
          stack[depth++] = uint32_t( node.child[1] );
          continue;
        }
      }

      if ( glm::dot( node.origin, receiver.normal ) <= receiver_dist + transfer_plane_epsilon )
        continue;

      const float trans = unnormalised_transfer( scene, receiver, node );
      if ( trans <= 0.0f )
        continue;

      if ( count < max_transfers )
      {
        if ( out_node )
        {
          out_node[count] = index;
          out_trans[count] = trans;
        }

        ++count;
      }
    }
  }

  return count;
}

inline bool transfer_visible( const transfer_scene& scene, const ray_tracer& tracer,
    uint32_t receiver_index, uint32_t shooter_index )
{
  const patch_node& receiver = scene.nodes[receiver_index];
  const patch_node& shooter = scene.nodes[shooter_index];
  const vec3 start = receiver.origin + receiver.normal;
  const vec3 delta = shooter.origin + shooter.normal - start;
  const float dist = glm::length( delta );

  return dist > ray_t_min &&
         !tracer.occluded( { start, delta / dist, dist - ray_t_min, geom::mask_world } );
}
