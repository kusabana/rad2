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

// utils/vrad/vrad_dispcoll.cpp:16
constexpr float triangle_edge_epsilon = 0.001f;

// utils/vrad/vrad_dispcoll.cpp:189
constexpr float grid_end_bias = 1.000001f;

// utils/vrad/vrad_dispcoll.cpp:178
struct disp_cell
{
  vec2 frac;
  size_t a;
  size_t b;
  size_t c;
  size_t d;
  bool odd;

  disp_cell( int32_t width, const vec2& uv )
  {
    const vec2 scaled = uv * ( float( width ) - grid_end_bias );
    const glm::ivec2 snap( scaled );
    const glm::ivec2 next = glm::min( snap + 1, width - 1 );
    frac = scaled - vec2( snap );
    a = size_t( snap.y * width + snap.x );
    b = size_t( snap.y * width + next.x );
    c = size_t( next.y * width + snap.x );
    d = size_t( next.y * width + next.x );

    odd = ( ( snap.y * width + snap.x ) & 1 ) != 0;
  }
};

struct disp_surface
{
  disp_surface( const scene_geometry& geometry, const face_info& info );

  vec3 point( const vec2& uv, vec3& triangle_normal ) const;
  vec3 normal( const vec2& uv ) const;

  vec3 point( const vec2& uv ) const
  {
    vec3 triangle_normal;
    return point( uv, triangle_normal );
  }

  int32_t width;
  std::span<const vec3> grid;
  std::vector<vec3> normals;
};

// public/builddisp.cpp:1732
disp_surface::disp_surface( const scene_geometry& geometry, const face_info& info )
    : width( ( 1 << info.disp_power ) + 1 ),
      grid( geometry.vertices.data() + info.disp_first_vertex, size_t( width ) * size_t( width ) ),
      normals( grid.size() )
{
  for ( int32_t i = 0; i < width; ++i )
  {
    for ( int32_t j = 0; j < width; ++j )
    {
      vec3 sum( 0.0f );
      int count = 0;

      for ( int32_t ci = glm::max( i - 1, 0 ); ci < glm::min( i + 1, width - 1 ); ++ci )
      {
        for ( int32_t cj = glm::max( j - 1, 0 ); cj < glm::min( j + 1, width - 1 ); ++cj )
        {
          const vec3& a = grid[size_t( ci * width + cj )];
          const vec3& b = grid[size_t( ci * width + cj + 1 )];
          const vec3& c = grid[size_t( ( ci + 1 ) * width + cj )];
          const vec3& d = grid[size_t( ( ci + 1 ) * width + cj + 1 )];
          sum += safe_normalize( glm::cross( b - a, c - a ) ) +
                 safe_normalize( glm::cross( d - b, c - b ) );
          count += 2;
        }
      }

      normals[size_t( i * width + j )] = sum / float( count );
    }
  }
}

// utils/vrad/vrad_dispcoll.cpp:178
vec3 disp_surface::point( const vec2& uv, vec3& triangle_normal ) const
{
  const disp_cell cell( width, uv );
  const vec3& a = grid[cell.a];
  const vec3& b = grid[cell.b];
  const vec3& c = grid[cell.c];
  const vec3& d = grid[cell.d];
  const vec2& f = cell.frac;

  if ( cell.odd )
  {
    if ( f.x + f.y >= 1.0f + triangle_edge_epsilon )
    {
      triangle_normal = safe_normalize( glm::cross( c - d, b - d ) );
      return d + ( c - d ) * ( 1.0f - f.x ) + ( b - d ) * ( 1.0f - f.y );
    }

    triangle_normal = safe_normalize( glm::cross( b - a, c - a ) );
    return a + ( b - a ) * f.x + ( c - a ) * f.y;
  }

  if ( f.x < f.y )
  {
    triangle_normal = safe_normalize( glm::cross( a - c, d - c ) );
    return c + ( d - c ) * f.x + ( a - c ) * ( 1.0f - f.y );
  }

  triangle_normal = safe_normalize( glm::cross( d - b, a - b ) );
  return b + ( a - b ) * ( 1.0f - f.x ) + ( d - b ) * f.y;
}

// utils/vrad/vrad_dispcoll.cpp:323
vec3 disp_surface::normal( const vec2& uv ) const
{
  const disp_cell cell( width, uv );
  const vec3 low =
      safe_normalize( normals[cell.a] * ( 1.0f - cell.frac.x ) + normals[cell.b] * cell.frac.x );
  const vec3 high =
      safe_normalize( normals[cell.c] * ( 1.0f - cell.frac.x ) + normals[cell.d] * cell.frac.x );
  return safe_normalize( low * ( 1.0f - cell.frac.y ) + high * cell.frac.y );
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

    if ( info.displacement() )
    {
      sample_displacement( geometry, uint32_t( face_index ) );
      continue;
    }

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

// utils/vrad/vraddisps.cpp:1559
void luxel_grid::sample_displacement( const scene_geometry& geometry, uint32_t face_index )
{
  const face_info& info = geometry.faces[face_index];
  const disp_surface surface( geometry, info );

  // public/builddisp.cpp:510
  const vec2 lightmap_end( float( info.width - 1 ), float( info.height - 1 ) );
  const vec2 luxel_to_uv = 1.0f / lightmap_end;

  for ( int32_t t = 0; t < info.height; ++t )
  {
    for ( int32_t s = 0; s < info.width; ++s )
    {
      const size_t sample_index =
          info.first_luxel + size_t( t ) * size_t( info.width ) + size_t( s );
      luxel& sample = samples[sample_index];
      sample.face_index = face_index;

      const vec2 cmins( s, t );
      const vec2 cmaxs = glm::min( cmins + 1.0f, lightmap_end );
      cell_mins[sample_index] = cmins;
      cell_maxs[sample_index] = cmaxs;
      sample.luxel_position = ( cmins + cmaxs ) * 0.5f;

      const vec2 uv = sample.luxel_position * luxel_to_uv;
      vec3 triangle_normal;
      const vec3 point = surface.point( uv, triangle_normal );

      // utils/vrad/vraddisps.cpp:1643
      sample.position = point + triangle_normal * sample_lift;
      sample.normal = surface.normal( uv );

      // utils/vrad/vraddisps.cpp:1617
      const vec3 p00 = surface.point( cmins * luxel_to_uv );
      const vec3 p10 = surface.point( vec2( cmaxs.x, cmins.y ) * luxel_to_uv );
      const vec3 p11 = surface.point( cmaxs * luxel_to_uv );
      const vec3 p01 = surface.point( vec2( cmins.x, cmaxs.y ) * luxel_to_uv );
      sample.world_area = 0.5f * ( glm::length( glm::cross( p10 - p00, p11 - p00 ) ) +
                                     glm::length( glm::cross( p11 - p00, p01 - p00 ) ) );
    }
  }
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

        // displaced samples can sit inside their own brush
        if ( face.model_index != 0 || face.displacement() )
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
