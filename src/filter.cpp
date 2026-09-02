#include "filter.h"

#include <algorithm>
#include <limits>
#include <unordered_map>

#include <tbb/parallel_for.h>

namespace
{

// constexpr float master_grid_coplanar_cos = const_cos( const_radians( 5.0f ) );
constexpr float master_grid_coplanar_cos = 0.9961947f;
constexpr float master_grid_plane_distance = 2.0f;

// utils/vrad/radial.cpp:89
constexpr float radial_ceil_bias = 0.9999f;

// utils/vrad/radial.cpp:115
constexpr float radial_min_distance = 0.1f;

// utils/vrad/lightmap.cpp:2920
constexpr float steep_gradient = 1.0f / 16.0f;

// utils/vrad/lightmap.cpp:2850
constexpr float perceptual_scale = byte_range;

// utils/vrad/radial.cpp:70
void splat_sample( const vec2& coord, const vec2& cmins, const vec2& cmaxs, const vec3& light,
    int32_t w, int32_t h, std::vector<vec3>& value, std::vector<float>& weight )
{
  const int32_t s_min = glm::max( int32_t( cmins.x ), 0 );
  const int32_t t_min = glm::max( int32_t( cmins.y ), 0 );
  const int32_t s_max = glm::min( int32_t( cmaxs.x + radial_ceil_bias ) + 1, int32_t( w ) - 1 );
  const int32_t t_max = glm::min( int32_t( cmaxs.y + radial_ceil_bias ) + 1, int32_t( h ) - 1 );

  for ( int32_t t = t_min; t <= t_max; ++t )
  {
    for ( int32_t s = s_min; s <= s_max; ++s )
    {
      const float s0 = glm::max( cmins.x - float( s ), -1.0f );
      const float t0 = glm::max( cmins.y - float( t ), -1.0f );
      const float s1 = glm::min( cmaxs.x - float( s ), 1.0f );
      const float t1 = glm::min( cmaxs.y - float( t ), 1.0f );
      const float area = ( s1 - s0 ) * ( t1 - t0 );
      if ( area <= equal_epsilon )
        continue;

      const float ds = glm::abs( coord.x - float( s ) );
      const float dt = glm::abs( coord.y - float( t ) );
      const float chebyshev = glm::max( glm::max( ds, dt ), radial_min_distance );

      const float r = area / chebyshev;
      const size_t i = size_t( t * w + s );
      value[i] += light * r;
      weight[i] += r;
    }
  }
}

} // namespace

// utils/vrad/lightmap.cpp:2907
std::vector<uint32_t> steep_gradient_luxels( const scene_geometry& geometry,
    const luxel_grid& luxels, const std::vector<vec3>& delta, std::vector<uint8_t>& processed )
{
  std::vector<float> intensity;
  std::vector<uint32_t> flagged;

  for ( const face_info& face : geometry.faces )
  {
    if ( !face.lightmapped )
      continue;

    const uint32_t first = face.first_luxel;
    const int32_t w = face.width;
    const int32_t h = face.height;

    intensity.assign( size_t( w ) * size_t( h ), 0.0f );

    for ( size_t i = 0; i < intensity.size(); ++i )
    {
      if ( luxels.samples[first + i].world_area <= 0.0f )
        continue;

      const vec3& v = delta[first + i];
      intensity[i] =
          glm::pow( ( v.x + v.y + v.z ) / perceptual_scale, 1.0f / float( light_gamma ) );
    }

    for ( int32_t t = 0; t < h; ++t )
    {
      for ( int32_t s = 0; s < w; ++s )
      {
        const size_t i = size_t( t * w + s );
        const uint32_t index = first + uint32_t( i );
        if ( luxels.samples[index].world_area <= 0.0f || processed[index] )
          continue;

        float gradient = 0.0f;

        for ( int32_t dt = -1; dt <= 1; ++dt )
        {
          for ( int32_t ds = -1; ds <= 1; ++ds )
          {
            const int32_t ns = s + ds;
            const int32_t nt = t + dt;
            if ( ns < 0 || ns >= w || nt < 0 || nt >= h )
              continue;

            const size_t j = size_t( nt * w + ns );
            if ( luxels.samples[first + j].world_area <= 0.0f )
              continue;

            gradient = glm::max( gradient, glm::abs( intensity[j] - intensity[i] ) );
          }
        }

        if ( gradient >= steep_gradient )
        {
          flagged.push_back( index );
          processed[index] = 1;
        }
      }
    }
  }

  return flagged;
}

// utils/vrad/radial.cpp:70
void radial_filter(
    const scene_geometry& geometry, const luxel_grid& luxels, std::vector<vec3>& delta )
{
  std::vector<vec3> filtered = delta;
  std::unordered_map<int64_t, std::vector<uint32_t>> groups;

  for ( uint32_t face_index = 0; face_index < geometry.faces.size(); ++face_index )
  {
    const face_info& face = geometry.faces[face_index];
    if ( !face.lightmapped )
      continue;

    const int64_t key =
        face.orig_face >= 0 ? int64_t( face.orig_face ) : -1 - int64_t( face_index );
    groups[key].push_back( face_index );
  }

  std::vector<std::vector<uint32_t>> group_list;
  group_list.reserve( groups.size() );

  for ( auto& [key, members] : groups )
    group_list.push_back( std::move( members ) );

  tbb::parallel_for( size_t( 0 ), group_list.size(),
      [&]( size_t gi )
      {
        const std::vector<uint32_t>& members = group_list[gi];
        const face_info& ref = geometry.faces[members[0]];

        std::vector<glm::ivec2> offsets( members.size() );
        int32_t m0 = std::numeric_limits<int32_t>::max();
        int32_t m1 = std::numeric_limits<int32_t>::max();
        int32_t x0 = std::numeric_limits<int32_t>::min();
        int32_t x1 = std::numeric_limits<int32_t>::min();

        for ( size_t mi = 0; mi < members.size(); ++mi )
        {
          const face_info& face = geometry.faces[members[mi]];
          const vec3 origin = face.luxel_to_world * vec3( 0.0f, 0.0f, 1.0f );
          const vec2 in_ref = world_to_st( ref.world_to_luxel_s, ref.world_to_luxel_t, origin );
          offsets[mi] =
              glm::ivec2( int32_t( glm::round( in_ref.x ) ), int32_t( glm::round( in_ref.y ) ) );
          m0 = glm::min( m0, offsets[mi].x );
          m1 = glm::min( m1, offsets[mi].y );
          x0 = glm::max( x0, offsets[mi].x + face.width - 1 );
          x1 = glm::max( x1, offsets[mi].y + face.height - 1 );
        }

        const int32_t w = x0 - m0 + 1;
        const int32_t h = x1 - m1 + 1;

        std::vector<vec3> value( size_t( w ) * size_t( h ) );
        std::vector<float> weight( size_t( w ) * size_t( h ) );

        for ( size_t mi = 0; mi < members.size(); ++mi )
        {
          const uint32_t fi = members[mi];
          const face_info& face = geometry.faces[fi];
          const vec2 off( float( offsets[mi].x - m0 ), float( offsets[mi].y - m1 ) );

          for ( size_t k = 0; k < face.luxel_count(); ++k )
          {
            const uint32_t index = face.first_luxel + uint32_t( k );
            const luxel& sample = luxels.samples[index];
            if ( sample.world_area <= 0.0f )
              continue;

            splat_sample( sample.luxel_position + off, luxels.cell_mins[index] + off,
                luxels.cell_maxs[index] + off, delta[index], w, h, value, weight );
          }
        }

        const vec2 ref_off( float( offsets[0].x - m0 ), float( offsets[0].y - m1 ) );

        std::vector<uint32_t> outside;

        for ( const uint32_t fi : members )
        {
          for ( uint32_t n = geometry.face_neighbor_first[fi];
              n < geometry.face_neighbor_first[fi + 1]; ++n )
          {
            const uint32_t neighbor_index = geometry.face_neighbors[n];
            const face_info& neighbor = geometry.faces[neighbor_index];
            if ( !neighbor.lightmapped )
              continue;

            if ( neighbor.orig_face >= 0 && neighbor.orig_face == ref.orig_face )
              continue;

            if ( glm::dot( ref.plane_normal, neighbor.plane_normal ) < master_grid_coplanar_cos )
              continue;

            const vec3 offset = luxels.samples[neighbor.first_luxel].position -
                                luxels.samples[ref.first_luxel].position;

            if ( glm::abs( glm::dot( ref.plane_normal, offset ) ) > master_grid_plane_distance )
              continue;

            if ( std::ranges::find( outside, neighbor_index ) == outside.end() )
              outside.push_back( neighbor_index );
          }
        }

        for ( const uint32_t neighbor_index : outside )
        {
          const face_info& neighbor = geometry.faces[neighbor_index];
          const size_t count = neighbor.luxel_count();

          for ( size_t k = 0; k < count; ++k )
          {
            const uint32_t index = neighbor.first_luxel + uint32_t( k );
            const luxel& sample = luxels.samples[index];
            if ( sample.world_area <= 0.0f )
              continue;

            const vec2 bmins = luxels.cell_mins[index];
            const vec2 bmaxs = luxels.cell_maxs[index];
            const vec3 world_min = neighbor.luxel_to_world * vec3( bmins.x, bmins.y, 1.0f );
            const vec3 world_max = neighbor.luxel_to_world * vec3( bmaxs.x, bmaxs.y, 1.0f );
            const vec2 coord = ref_off + world_to_st( ref.world_to_luxel_s, ref.world_to_luxel_t,
                                             sample.position );
            const vec2 c0 =
                ref_off + world_to_st( ref.world_to_luxel_s, ref.world_to_luxel_t, world_min );
            const vec2 c1 =
                ref_off + world_to_st( ref.world_to_luxel_s, ref.world_to_luxel_t, world_max );
            splat_sample(
                coord, glm::min( c0, c1 ), glm::max( c0, c1 ), delta[index], w, h, value, weight );
          }
        }

        for ( size_t mi = 0; mi < members.size(); ++mi )
        {
          const uint32_t fi = members[mi];
          const face_info& face = geometry.faces[fi];
          const int32_t offx = offsets[mi].x - m0;
          const int32_t offy = offsets[mi].y - m1;

          for ( size_t k = 0; k < face.luxel_count(); ++k )
          {
            const uint32_t index = face.first_luxel + uint32_t( k );
            const size_t mk = size_t( int32_t( k / size_t( face.width ) ) + offy ) * size_t( w ) +
                              size_t( int32_t( k % size_t( face.width ) ) + offx );

            if ( weight[mk] > radial_weight_eps )
            {
              filtered[index] = value[mk] / weight[mk];
            }
            else
            {
              filtered[index] = vec3( 0.0f );
            }
          }
        }
      } );

  delta.swap( filtered );
}
