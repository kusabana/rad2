#include "geometry.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <limits>
#include <span>
#include <unordered_map>
#include <unordered_set>

namespace
{

// utils/vrad/lightmap.cpp:32
constexpr uint32_t smoothing_group_hard_edge = 0xff000000;
constexpr int32_t non_lit_surf_flags =
    surf::sky | surf::sky2d | surf::nolight | surf::nodraw | surf::hint | surf::skip;

// utils/vrad/lightmap.cpp:123
uint32_t face_vertex( const bsp_file& bsp, const dface& face, int32_t edge )
{
  const int32_t surfedge = bsp.surfedges[size_t( face.first_edge + edge )];
  if ( surfedge < 0 )
    return bsp.edges[size_t( -surfedge )].vertex[1];

  return bsp.edges[size_t( surfedge )].vertex[0];
}

void emit_polygon( std::span<const uint32_t> polygon_vertices, std::vector<uint32_t>& indices )
{
  for ( size_t k = 2; k < polygon_vertices.size(); ++k )
    indices.insert(
        indices.end(), { polygon_vertices[0], polygon_vertices[k - 1], polygon_vertices[k] } );
}

// utils/common/polylib.cpp:296
constexpr float brush_winding_half_size = 4.0f * max_coord_integer;
constexpr float brush_chop_epsilon = on_epsilon;

// windings grow this much so neighbouring brushes overlap
constexpr float occluder_margin = 0.25f;

constexpr float corner_limit = 0.5f;

// utils/vrad/vrad.cpp:420
constexpr float max_reflectivity = 0.99f;

// public/bspfile.h:32
constexpr int32_t max_lightmap_width = 35;
// public/bspfile.h:36
constexpr int32_t max_disp_lightmap_width = 128;

// public/bspfile.h:47
constexpr int32_t min_disp_power = 2;
constexpr int32_t max_disp_power = 4;

// utils/vrad/vrad.h:542
bool valid_displacement( const bsp_file& bsp, const dface& face )
{
  if ( face.dispinfo < 0 || size_t( face.dispinfo ) >= bsp.dispinfos.size() || face.num_edges != 4 )
    return false;

  const ddispinfo& disp = bsp.dispinfos[size_t( face.dispinfo )];
  if ( disp.power < min_disp_power || disp.power > max_disp_power || disp.disp_vert_start < 0 )
    return false;

  const size_t width = size_t( 1 << disp.power ) + 1;
  return size_t( disp.disp_vert_start ) + width * width <= bsp.disp_verts.size();
}

constexpr float phong_epsilon = 1.0e-12f;

// vertex normals dedup on a fixed point key
constexpr int normal_key_bits = 64 / 3;
constexpr int normal_key_fraction_bits = 14;
constexpr float normal_key_scale = float( 1 << normal_key_fraction_bits );
constexpr uint64_t normal_key_mask = ( uint64_t( 1 ) << normal_key_bits ) - 1;

void base_winding( const vec3& normal, float dist, std::vector<vec3>& out )
{
  const vec3 abs_n = glm::abs( normal );
  vec3 axis;

  if ( abs_n.x > abs_n.y && abs_n.x > abs_n.z )
    axis = vec3( 0.0f, 1.0f, 0.0f );
  else if ( abs_n.y > abs_n.z )
    axis = vec3( 0.0f, 0.0f, 1.0f );
  else
    axis = vec3( 1.0f, 0.0f, 0.0f );

  const vec3 tangent = safe_normalize( axis - normal * glm::dot( normal, axis ) );
  const vec3 bitangent = glm::cross( normal, tangent );
  const vec3 center = normal * dist;

  out.clear();
  out.push_back( center + tangent * brush_winding_half_size + bitangent * brush_winding_half_size );
  out.push_back( center - tangent * brush_winding_half_size + bitangent * brush_winding_half_size );
  out.push_back( center - tangent * brush_winding_half_size - bitangent * brush_winding_half_size );
  out.push_back( center + tangent * brush_winding_half_size - bitangent * brush_winding_half_size );
}

// push every edge of a convex winding outward by occluder_margin
void grow_winding( std::vector<vec3>& winding, const vec3& plane_normal )
{
  const size_t count = winding.size();
  if ( count < 3 )
    return;

  std::vector<vec3> outward( count );

  for ( size_t k = 0; k < count; ++k )
  {
    const vec3 edge = winding[( k + 1 ) % count] - winding[k];
    const vec3 normal = glm::cross( edge, plane_normal );
    const float length = glm::length( normal );
    outward[k] = length > length_epsilon ? normal / length : vec3( 0.0f );
  }

  for ( size_t k = 0; k < count; ++k )
  {
    const vec3& before = outward[( k + count - 1 ) % count];
    const vec3& after = outward[k];
    const float denominator = glm::max( 1.0f + glm::dot( before, after ), corner_limit );
    winding[k] += ( before + after ) * ( occluder_margin / denominator );
  }
}

std::unordered_set<uint32_t> collect_model_brushes( const bsp_file& bsp, int32_t head_node )
{
  std::unordered_set<uint32_t> out;

  std::vector<int32_t> stack;
  stack.push_back( head_node );

  while ( !stack.empty() )
  {
    const int32_t child = stack.back();
    stack.pop_back();

    if ( child >= 0 )
    {
      stack.push_back( bsp.nodes[child].children[0] );
      stack.push_back( bsp.nodes[child].children[1] );
      continue;
    }

    const dleaf_v1& leaf = bsp.leaves[size_t( -child - 1 )];
    const size_t first = leaf.first_leaf_brush;

    for ( size_t i = 0; i < leaf.num_leaf_brushes; ++i )
      out.insert( bsp.leaf_brushes[first + i] );
  }

  return out;
}

void collect_brush_occluders( const bsp_file& bsp, const std::unordered_set<uint32_t>& brush_set,
    std::vector<vec3>& pool, std::vector<uint32_t>& indices )
{
  std::vector<vec3> winding;
  std::vector<vec3> scratch;
  std::vector<uint32_t> polygon;

  for ( const uint32_t brush_index : brush_set )
  {
    const dbrush& brush = bsp.brushes[brush_index];
    if ( ( uint32_t( brush.contents ) & contents::mask_opaque ) == 0 )
      continue;

    const size_t first = size_t( brush.first_side );
    const size_t count = size_t( brush.num_sides );

    for ( size_t i = 0; i < count; ++i )
    {
      const dbrushside& side = bsp.brushsides[first + i];
      if ( side.bevel != 0 || side.dispinfo != 0 )
        continue;

      int32_t side_flags = 0;
      if ( side.texinfo >= 0 && size_t( side.texinfo ) < bsp.texinfos.size() )
        side_flags = bsp.texinfos[side.texinfo].flags;

      if ( side_flags & surf::sky )
        continue;

      const dplane& plane = bsp.planes[side.plane_num];
      base_winding( plane.normal, plane.dist, winding );

      for ( size_t j = 0; j < count && winding.size() >= 3; ++j )
      {
        if ( i == j )
          continue;

        const dbrushside& other = bsp.brushsides[first + j];
        if ( other.bevel != 0 )
          continue;

        const dplane& other_plane = bsp.planes[other.plane_num];
        clip_convex( winding, -other_plane.normal, -other_plane.dist, brush_chop_epsilon, scratch );
        std::swap( winding, scratch );
      }

      if ( winding.size() < 3 )
        continue;

      grow_winding( winding, plane.normal );
      polygon.clear();

      for ( const vec3& p : winding )
      {
        polygon.push_back( uint32_t( pool.size() ) );
        pool.push_back( p );
      }

      emit_polygon( polygon, indices );
    }
  }
}

} // namespace

// utils/vrad/vrad_dispcoll.cpp:1064
void scene_geometry::collect_displacement_occluders( const bsp_file& bsp )
{
  for ( size_t face_index = 0; face_index < faces.size(); ++face_index )
  {
    face_info& info = faces[face_index];
    if ( !info.displacement() )
      continue;

    const ddispinfo& disp = bsp.dispinfos[size_t( bsp.faces[face_index].dispinfo )];
    const std::span<const vec3> corners( face_vertex_position.data() + info.first_vertex, 4 );

    // the grid starts nearest the stored start position
    const auto nearest = std::ranges::min_element( corners, {}, [&]( const vec3& corner )
        { return glm::length( corner - info.model_origin - disp.start_position ); } );
    const size_t start_corner = size_t( nearest - corners.begin() );
    std::array<vec3, 4> points;

    for ( size_t j = 0; j < 4; ++j )
      points[j] = corners[( start_corner + j ) & 3];

    // public/builddisp.cpp:1918
    const int32_t post_spacing = ( 1 << disp.power ) + 1;
    const float oo_int = 1.0f / float( post_spacing - 1 );
    const std::array<vec3, 2> edge_int = {
        ( points[1] - points[0] ) * oo_int, ( points[2] - points[3] ) * oo_int };
    const uint32_t base = uint32_t( vertices.size() );
    info.disp_first_vertex = int32_t( base );

    for ( int32_t i = 0; i < post_spacing; ++i )
    {
      const vec3 end0 = points[0] + edge_int[0] * float( i );
      const vec3 end1 = points[3] + edge_int[1] * float( i );
      const vec3 seg_int = ( end1 - end0 ) * oo_int;

      for ( int32_t j = 0; j < post_spacing; ++j )
      {
        const ddispvert& v =
            bsp.disp_verts[size_t( disp.disp_vert_start ) + size_t( i ) * post_spacing + j];
        vertices.push_back( end0 + seg_int * float( j ) + v.vec * v.dist );
      }
    }

    for ( int32_t i = 0; i + 1 < post_spacing; ++i )
    {
      for ( int32_t j = 0; j + 1 < post_spacing; ++j )
      {
        const uint32_t a = base + uint32_t( i * post_spacing + j );
        const uint32_t b = a + 1;
        const uint32_t c = a + uint32_t( post_spacing );
        const uint32_t d = c + 1;

        if ( ( i * post_spacing + j ) & 1 )
        {
          occluder_indices.insert( occluder_indices.end(), { a, b, c } );
          occluder_indices.insert( occluder_indices.end(), { b, d, c } );
        }
        else
        {
          occluder_indices.insert( occluder_indices.end(), { a, b, d } );
          occluder_indices.insert( occluder_indices.end(), { a, d, c } );
        }
      }
    }
  }
}

void scene_geometry::build_face_infos( const bsp_file& bsp, const std::vector<int32_t>& face_model,
    const std::vector<vec3>& model_origin )
{
  faces.resize( bsp.faces.size() );
  face_centroid_.resize( bsp.faces.size() );
  face_vertex_position.reserve( bsp.surfedges.size() );
  face_vertex_index_.reserve( bsp.surfedges.size() );

  uint64_t luxel_total = 0;

  for ( size_t face_index = 0; face_index < bsp.faces.size(); ++face_index )
  {
    const dface& face = bsp.faces[face_index];
    face_info& info = faces[face_index];

    info.first_luxel = uint32_t( luxel_total );
    info.model_index = face_model[face_index];
    info.model_origin = model_origin[info.model_index];
    info.orig_face = face.orig_face;
    info.first_vertex = uint32_t( face_vertex_position.size() );
    info.vertex_count = uint32_t( glm::max<int16_t>( face.num_edges, 0 ) );

    const dplane& plane = bsp.planes[face.plane_num];
    info.plane_normal = plane.normal;
    info.plane_dist = plane.dist + glm::dot( info.model_origin, info.plane_normal );

    vec3 centroid( 0.0f );

    for ( int32_t j = 0; j < face.num_edges; ++j )
    {
      const uint32_t vertex = face_vertex( bsp, face, j );
      const vec3 position = bsp.vertices[vertex].position + info.model_origin;
      face_vertex_index_.push_back( vertex );
      face_vertex_position.push_back( position );
      centroid += position;
    }

    if ( face.num_edges > 0 )
      face_centroid_[face_index] = centroid / float( face.num_edges );

    if ( face.texinfo < 0 || size_t( face.texinfo ) >= bsp.texinfos.size() )
      continue;

    const dtexinfo& texinfo = bsp.texinfos[face.texinfo];

    info.surf_flags = texinfo.flags;
    if ( valid_displacement( bsp, face ) )
      info.disp_power = bsp.dispinfos[size_t( face.dispinfo )].power;

    if ( texinfo.texdata >= 0 && size_t( texinfo.texdata ) < bsp.texdatas.size() )
      info.reflectivity =
          glm::min( bsp.texdatas[texinfo.texdata].reflectivity, vec3( max_reflectivity ) );

    if ( texinfo.flags & non_lit_surf_flags )
      continue;

    if ( face.num_edges < 3 )
      continue;

    if ( face.dispinfo >= 0 && !info.displacement() )
      continue;

    info.width = face.lightmap_size[0] + 1;
    info.height = face.lightmap_size[1] + 1;

    const int32_t max_width = info.displacement() ? max_disp_lightmap_width : max_lightmap_width;
    if ( info.width <= 0 || info.height <= 0 || info.width > max_width || info.height > max_width )
      continue;

    // utils/vrad/lightmap.cpp:450
    const glm::dvec3 s_vec( vec3( texinfo.lightmap_vecs[0] ) );
    const glm::dvec3 t_vec( vec3( texinfo.lightmap_vecs[1] ) );
    const glm::dvec3 normal( info.plane_normal );

    const glm::dvec3 cross_st = glm::cross( t_vec, s_vec );
    const double det = -glm::dot( normal, cross_st );
    if ( glm::abs( det ) < basis_epsilon )
    {
      log_warn( "face {} has lightmap vectors parallel to its normal", face_index );
      continue;
    }

    const double inv_det = 1.0 / det;
    const glm::dvec3 s_axis = glm::cross( t_vec, normal ) * inv_det;
    const glm::dvec3 t_axis = glm::cross( normal, s_vec ) * inv_det;

    const double s_offset = double( texinfo.lightmap_vecs[0].w ) - face.lightmap_mins[0];
    const double t_offset = double( texinfo.lightmap_vecs[1].w ) - face.lightmap_mins[1];

    const double base_dist = double( plane.dist );
    const glm::dvec3 origin =
        cross_st * ( -base_dist * inv_det ) - s_axis * s_offset - t_axis * t_offset;

    info.luxel_to_world =
        mat3( vec3( s_axis ), vec3( t_axis ), vec3( origin ) + info.model_origin );

    info.world_to_luxel_s = vec4(
        vec3( s_vec ), float( s_offset - glm::dot( glm::dvec3( info.model_origin ), s_vec ) ) );
    info.world_to_luxel_t = vec4(
        vec3( t_vec ), float( t_offset - glm::dot( glm::dvec3( info.model_origin ), t_vec ) ) );

    info.lightmapped = true;
    luxel_total += uint64_t( info.width ) * uint64_t( info.height );
  }

  if ( luxel_total > uint64_t( std::numeric_limits<int32_t>::max() ) )
    throw fatal_error( "map exceeds 2^31 luxels" );

  luxel_count = uint32_t( luxel_total );
}

void scene_geometry::build_edge_planes()
{
  std::vector<vec2> winding;

  for ( face_info& info : faces )
  {
    if ( !info.lightmapped )
      continue;

    winding.clear();

    for ( uint32_t j = 0; j < info.vertex_count; ++j )
    {
      winding.push_back( world_to_st( info.world_to_luxel_s, info.world_to_luxel_t,
          face_vertex_position[info.first_vertex + j] ) );
    }

    const float orient = ( signed_area2( winding ) >= 0.0f ) ? 1.0f : -1.0f;

    info.edge_first = uint32_t( edge_planes.size() );

    for ( size_t i = 0; i < winding.size(); ++i )
    {
      const vec2& a = winding[i];
      const std::optional<vec2> inward =
          inward_edge_normal( a, winding[( i + 1 ) % winding.size()], orient );

      if ( inward )
        edge_planes.push_back( { *inward, glm::dot( *inward, a ) } );
    }

    info.edge_count = uint32_t( edge_planes.size() ) - info.edge_first;
  }
}

// utils/vrad/trace.cpp:612
void scene_geometry::emit_sky_triangles()
{
  vertices = face_vertex_position;

  std::vector<uint32_t> polygon;

  for ( size_t face_index = 0; face_index < faces.size(); ++face_index )
  {
    const face_info& info = faces[face_index];
    if ( info.model_index != 0 || !( info.surf_flags & ( surf::sky | surf::sky2d ) ) )
      continue;

    polygon.clear();

    for ( uint32_t j = 0; j < info.vertex_count; ++j )
      polygon.push_back( info.first_vertex + j );

    emit_polygon( polygon, sky_indices );
  }
}

void scene_geometry::build_smoothed_normals( const bsp_file& bsp )
{
  std::vector<std::vector<uint32_t>> vertex_faces( bsp.vertices.size() );

  for ( size_t face_index = 0; face_index < bsp.faces.size(); ++face_index )
  {
    const dface& face = bsp.faces[face_index];
    if ( face.dispinfo >= 0 )
      continue;

    const face_info& info = faces[face_index];

    for ( uint32_t j = 0; j < info.vertex_count; ++j )
    {
      std::vector<uint32_t>& list = vertex_faces[face_vertex_index_[info.first_vertex + j]];
      if ( std::ranges::find( list, uint32_t( face_index ) ) == list.end() )
        list.push_back( uint32_t( face_index ) );
    }
  }

  face_vertex_normal_.resize( face_vertex_position.size() );

  face_neighbor_first.reserve( bsp.faces.size() + 1 );
  std::vector<uint32_t> neighbors;

  for ( size_t face_index = 0; face_index < bsp.faces.size(); ++face_index )
  {
    const dface& face = bsp.faces[face_index];
    const face_info& info = faces[face_index];

    face_neighbor_first.push_back( uint32_t( face_neighbors.size() ) );
    neighbors.clear();

    if ( face.dispinfo >= 0 )
      continue;

    for ( uint32_t j = 0; j < info.vertex_count; ++j )
    {
      vec3 sum( 0.0f );

      for ( const uint32_t neighbor_index :
          vertex_faces[face_vertex_index_[info.first_vertex + j]] )
      {
        if ( neighbor_index == face_index )
          continue;

        const dface& neighbor = bsp.faces[neighbor_index];
        const face_info& neighbor_info = faces[neighbor_index];
        const float cos_angle = glm::dot( neighbor_info.plane_normal, info.plane_normal );

        if ( face.smoothing_groups == 0 && neighbor.smoothing_groups == 0 )
        {
          if ( cos_angle < smooth_cos_threshold )
            continue;
        }
        else
        {
          const uint32_t group = face.smoothing_groups & neighbor.smoothing_groups;
          if ( group & smoothing_group_hard_edge )
            continue;

          if ( group == 0 )
            continue;
        }

        sum += neighbor_info.plane_normal * glm::max( glm::min( neighbor.area, face.area ), 0.0f );

        if ( std::ranges::find( neighbors, neighbor_index ) == neighbors.end() )
        {
          neighbors.push_back( neighbor_index );
          face_neighbors.push_back( neighbor_index );
        }
      }

      sum += info.plane_normal * glm::max( face.area, 0.0f );
      const float length = glm::length( sum );
      face_vertex_normal_[info.first_vertex + j] =
          length > length_epsilon ? sum / length : info.plane_normal;
    }
  }

  face_neighbor_first.push_back( uint32_t( face_neighbors.size() ) );
}

// utils/vrad/lightmap.cpp:328
void scene_geometry::dedup_vertex_normals()
{
  std::unordered_map<uint64_t, uint16_t> dedup;
  vertex_normal_indices.reserve( face_vertex_normal_.size() );

  const auto quantize = []( float v )
  {
    return uint64_t( int32_t( glm::round( v * normal_key_scale ) ) ) & normal_key_mask;
  };

  for ( const vec3& normal : face_vertex_normal_ )
  {
    const uint64_t key = quantize( normal.x ) | ( quantize( normal.y ) << normal_key_bits ) |
                         ( quantize( normal.z ) << ( 2 * normal_key_bits ) );

    const auto [slot, added] = dedup.try_emplace( key, uint16_t( vertex_normal_table.size() ) );
    if ( added )
    {
      if ( vertex_normal_table.size() >= 0xFFFF )
        throw fatal_error( "more than 65535 unique vertex normals" );

      vertex_normal_table.push_back( normal );
    }

    vertex_normal_indices.push_back( slot->second );
  }
}

scene_geometry::scene_geometry( const bsp_file& bsp, const entity_lump& entities )
{
  if ( bsp.models.empty() )
    throw fatal_error( "map has no models" );

  std::vector<int32_t> face_model( bsp.faces.size() );
  std::vector<vec3> model_origin( bsp.models.size() );

  for ( size_t model_index = 0; model_index < bsp.models.size(); ++model_index )
  {
    const dmodel& model = bsp.models[model_index];

    for ( int32_t i = 0; i < model.num_faces; ++i )
    {
      const size_t face_index = size_t( model.first_face ) + size_t( i );
      if ( face_index < face_model.size() )
        face_model[face_index] = int32_t( model_index );
    }
  }

  for ( const entity& e : entities.entities() )
  {
    const std::optional<std::string_view> model_value = e.value( "model" );
    if ( !model_value || model_value->size() < 2 || ( *model_value )[0] != '*' )
      continue;

    const std::string_view digits = model_value->substr( 1 );
    int32_t model_index = 0;
    if ( std::from_chars( digits.data(), digits.data() + digits.size(), model_index ).ec !=
             std::errc() ||
         model_index <= 0 || size_t( model_index ) >= bsp.models.size() )
      continue;

    model_origin[model_index] = e.vec3_value( "origin" );
  }

  build_face_infos( bsp, face_model, model_origin );
  build_edge_planes();
  emit_sky_triangles();

  collect_brush_occluders(
      bsp, collect_model_brushes( bsp, bsp.models[0].head_node ), vertices, occluder_indices );
  collect_displacement_occluders( bsp );

  build_smoothed_normals( bsp );
  dedup_vertex_normals();
}

// utils/vrad/lightmap.cpp:2118
vec3 scene_geometry::phong_normal( uint32_t face_index, const vec3& spot ) const
{
  const face_info& info = faces[face_index];
  vec3 result = info.plane_normal;

  if ( info.vertex_count < 3 )
    return result;

  const vec3 centroid = face_centroid_[face_index];

  for ( uint32_t j = 0; j < info.vertex_count; ++j )
  {
    const uint32_t j_next = ( j + 1 ) % info.vertex_count;
    const vec3& n1 = face_vertex_normal_[info.first_vertex + j];
    const vec3& n2 = face_vertex_normal_[info.first_vertex + j_next];

    const vec3 v1 = face_vertex_position[info.first_vertex + j] - centroid;
    const vec3 v2 = face_vertex_position[info.first_vertex + j_next] - centroid;
    const vec3 vspot = spot - centroid;

    const float aa = glm::dot( v1, v1 );
    const float bb = glm::dot( v2, v2 );
    const float ab = glm::dot( v1, v2 );
    const float denom = aa * bb - ab * ab;
    if ( glm::abs( denom ) < phong_epsilon )
      continue;

    const float a1 = ( bb * glm::dot( v1, vspot ) - ab * glm::dot( vspot, v2 ) ) / denom;
    const float a2 = ( glm::dot( vspot, v2 ) - a1 * ab ) / bb;
    if ( a1 >= 0.0f && a2 >= 0.0f )
    {
      const float scale = 1.0f - a1 - a2;
      result = info.plane_normal * scale + n1 * a1 + n2 * a2;

      const float length = glm::length( result );
      if ( length > length_epsilon )
        return result / length;

      return info.plane_normal;
    }
  }

  return result;
}
