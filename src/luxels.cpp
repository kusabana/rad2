#include "luxels.h"

#include <tbb/parallel_for.h>

namespace
{

constexpr float min_cell_area = 1.0e-5f;
constexpr float area_epsilon = 1.0e-12f;

float polygon_area_and_centroid( std::span<const vec2> points, vec2& centroid )
{
  const float area2 = signed_area2( points );
  if ( glm::abs( area2 ) < area_epsilon )
    return 0.0f;

  vec2 weighted( 0.0f );

  for ( size_t i = 0; i < points.size(); ++i )
  {
    const vec2& a = points[i];
    const vec2& b = points[( i + 1 ) % points.size()];
    weighted += ( a + b ) * ( a.x * b.y - b.x * a.y );
  }

  centroid = weighted / ( 3.0f * area2 );
  return area2 * 0.5f;
}

} // namespace

luxel_grid::luxel_grid( const bsp_file& bsp, const scene_geometry& geometry )
{
  const size_t total_luxels = geometry.luxel_count;

  samples.resize( total_luxels );
  cell_mins.resize( total_luxels );
  cell_maxs.resize( total_luxels );

  std::vector<vec2> winding;
  std::vector<vec2> cell;
  std::vector<vec2> scratch;

  for ( size_t face_index = 0; face_index < geometry.faces.size(); ++face_index )
  {
    const face_info& info = geometry.faces[face_index];
    if ( !info.lightmapped )
      continue;

    winding.clear();

    for ( uint32_t j = 0; j < info.vertex_count; ++j )
    {
      winding.push_back( world_to_st( info.world_to_luxel_s, info.world_to_luxel_t,
          geometry.face_vertex_position[info.first_vertex + j] ) );
    }

    const float world_area_per_luxel =
        1.0f / glm::max( glm::length( vec3( info.world_to_luxel_s ) ) *
                             glm::length( vec3( info.world_to_luxel_t ) ),
                   area_epsilon );

    for ( int32_t t = 0; t < info.height; ++t )
    {
      for ( int32_t s = 0; s < info.width; ++s )
      {
        const size_t sample_index = info.first_luxel + size_t( t ) * info.width + s;
        luxel& sample = samples[sample_index];

        sample.face_index = uint32_t( face_index );

        const float s0 = float( s );
        const float s1 = s0 + 1.0f;
        const float t0 = float( t );
        const float t1 = t0 + 1.0f;

        clip_convex( winding, vec2( 1.0f, 0.0f ), s0, 0.0f, cell );
        clip_convex( cell, vec2( -1.0f, 0.0f ), -s1, 0.0f, scratch );
        clip_convex( scratch, vec2( 0.0f, 1.0f ), t0, 0.0f, cell );
        clip_convex( cell, vec2( 0.0f, -1.0f ), -t1, 0.0f, scratch );

        vec2 centroid( s0 + 0.5f, t0 + 0.5f );
        const float cell_area = glm::abs( polygon_area_and_centroid( scratch, centroid ) );
        sample.luxel_position = centroid;

        vec2 cmins( s0 + 0.5f, t0 + 0.5f );
        vec2 cmaxs = cmins;

        for ( const vec2& v : scratch )
        {
          cmins = glm::min( cmins, v );
          cmaxs = glm::max( cmaxs, v );
        }

        cell_mins[sample_index] = cmins;
        cell_maxs[sample_index] = cmaxs;

        if ( cell_area > min_cell_area )
          sample.world_area = cell_area * world_area_per_luxel;
        else
          sample.world_area = 0.0f;

        sample.position =
            info.luxel_to_world * vec3( sample.luxel_position.x, sample.luxel_position.y, 1.0f );
        sample.normal = geometry.phong_normal( uint32_t( face_index ), sample.position );
      }
    }
  }

  find_leaves( geometry, bsp );
}

namespace
{

constexpr float inside_margin = 0.5f;

} // namespace

void luxel_grid::find_leaves( const scene_geometry& geometry, const bsp_file& bsp )
{
  const auto inside_brush = [&]( const dbrush& brush, const vec3& point )
  {
    if ( ( uint32_t( brush.contents ) & contents::mask_opaque ) == 0 )
      return false;

    for ( int32_t s = 0; s < brush.num_sides; ++s )
    {
      const dplane& plane = bsp.planes[bsp.brushsides[size_t( brush.first_side ) + s].plane_num];
      if ( glm::dot( plane.normal, point ) - plane.dist > -inside_margin )
        return false;
    }

    return true;
  };

  tbb::parallel_for( size_t( 0 ), samples.size(),
      [&]( size_t i )
      {
        luxel& sample = samples[i];
        const face_info& face = geometry.faces[sample.face_index];
        if ( sample.world_area <= 0.0f )
          return;

        const vec3 point = sample.position + face.plane_normal * sample_lift;
        const dleaf_v1& leaf = bsp.leaf_at( point );
        sample.area = leaf.area();
        if ( face.model_index != 0 )
          return;

        for ( uint16_t b = 0; b < leaf.num_leaf_brushes; ++b )
        {
          const size_t brush_index = bsp.leaf_brushes[size_t( leaf.first_leaf_brush ) + b];
          if ( inside_brush( bsp.brushes[brush_index], point ) )
          {
            sample.world_area = 0.0f;
            break;
          }
        }
      } );
}
