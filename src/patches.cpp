#include "patches.h"
#include "filter.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <numeric>
#include <tbb/parallel_for.h>

namespace
{

// utils/vrad/vrad.cpp:53
constexpr float chop = 4.0f;
constexpr float clip_epsilon = on_epsilon;

// utils/common/polylib.h:30
constexpr int32_t max_winding_points = 64;

// utils/common/polylib.cpp:106
constexpr float collinear_cos = 1.0f - equal_epsilon;

// utils/common/polylib.cpp:90
void remove_colinear_points( std::vector<vec3>& winding )
{
  std::vector<vec3> kept;

  for ( size_t i = 0; i < winding.size(); ++i )
  {
    const size_t j = ( i + 1 ) % winding.size();
    const size_t k = ( i + winding.size() - 1 ) % winding.size();
    const vec3 v1 = safe_normalize( winding[j] - winding[i] );
    const vec3 v2 = safe_normalize( winding[i] - winding[k] );
    if ( glm::dot( v1, v2 ) < collinear_cos )
      kept.push_back( winding[i] );
  }

  winding = std::move( kept );
}

// utils/common/polylib.cpp:197
vec3 winding_center( std::span<const vec3> winding )
{
  vec3 center( 0.0f );

  for ( const vec3& point : winding )
    center += point;

  return center * ( 1.0f / float( winding.size() ) );
}

struct winding_measure
{
  float area;
  vec3 balance;
};

winding_measure measure_winding( std::span<const vec3> winding )
{
  winding_measure result{};
  float total = 0.0f;

  for ( size_t i = 2; i < winding.size(); ++i )
  {
    const float area =
        glm::length( glm::cross( winding[i - 1] - winding[0], winding[i] - winding[0] ) );
    total += area;
    result.balance += ( winding[i - 1] + winding[i] + winding[0] ) * ( area / 3.0f );
  }

  if ( total != 0.0f )
    result.balance *= 1.0f / total;

  result.area = total * 0.5f;
  return result;
}

// utils/common/polylib.cpp:364
void clip_winding_epsilon( const std::vector<vec3>& in, int axis, float dist,
    std::vector<vec3>& front, std::vector<vec3>& back )
{
  front.clear();
  back.clear();

  const size_t count = in.size();
  std::array<float, max_winding_points + 4> dists;
  std::array<int, max_winding_points + 4> sides;
  std::array<int, 3> counts = { 0, 0, 0 };

  for ( size_t i = 0; i < count; ++i )
  {
    const float dot = in[i][axis] - dist;
    dists[i] = dot;

    if ( dot > clip_epsilon )
      sides[i] = 0;
    else if ( dot < -clip_epsilon )
      sides[i] = 1;
    else
      sides[i] = 2;

    ++counts[sides[i]];
  }

  sides[count] = sides[0];
  dists[count] = dists[0];

  if ( counts[0] == 0 )
  {
    back = in;
    return;
  }

  if ( counts[1] == 0 )
  {
    front = in;
    return;
  }

  for ( size_t i = 0; i < count; ++i )
  {
    const vec3& p1 = in[i];

    if ( sides[i] == 2 )
    {
      front.push_back( p1 );
      back.push_back( p1 );
      continue;
    }

    if ( sides[i] == 0 )
      front.push_back( p1 );
    else
      back.push_back( p1 );

    if ( sides[i + 1] == 2 || sides[i + 1] == sides[i] )
      continue;

    const vec3& p2 = in[( i + 1 ) % count];
    const float dot = dists[i] / ( dists[i] - dists[i + 1] );
    vec3 mid = p1 + ( p2 - p1 ) * dot;
    mid[axis] = dist;
    front.push_back( mid );
    back.push_back( mid );
  }
}

struct leaf_bounds
{
  uint32_t node;
  vec3 mins;
  vec3 maxs;
};

} // namespace

struct patch_tree::builder
{
  builder( const scene_geometry& geometry, patch_tree& tree ) : geometry( geometry ), tree( tree )
  {
  }

  const scene_geometry& geometry;
  patch_tree& tree;
  std::vector<vec3> front;
  std::vector<vec3> back;

  uint32_t add_node( const std::vector<vec3>& points, const patch_node& proto )
  {
    patch_node node = proto;
    node.winding_first = uint32_t( tree.winding.size() );
    node.winding_count = uint32_t( points.size() );
    tree.winding.insert( tree.winding.end(), points.begin(), points.end() );
    tree.nodes.push_back( node );
    return uint32_t( tree.nodes.size() - 1 );
  }

  // utils/vrad/vrad.cpp:835
  void subdivide( uint32_t index, float luxscale )
  {
    const patch_node node = tree.nodes[index];
    if ( node.sky )
      return;

    std::vector<vec3> points( tree.winding.begin() + node.winding_first,
        tree.winding.begin() + node.winding_first + node.winding_count );

    vec3 mins( float_max );
    vec3 maxs( float_lowest );

    for ( const vec3& point : points )
    {
      mins = glm::min( mins, point );
      maxs = glm::max( maxs, point );
    }

    const vec3 total = ( maxs - mins ) * luxscale;
    int widest_axis = -1;
    float widest = -1.0f;
    bool split = false;

    for ( int i = 0; i < 3; ++i )
    {
      if ( total[i] > widest )
      {
        widest_axis = i;
        widest = total[i];
      }

      if ( total[i] >= chop )
        split = true;
    }

    if ( !split || widest_axis < 0 )
      return;

    clip_winding_epsilon(
        points, widest_axis, 0.5f * ( mins[widest_axis] + maxs[widest_axis] ), front, back );

    const winding_measure front_measure = measure_winding( front );
    const winding_measure back_measure = measure_winding( back );
    if ( front_measure.area == 0.0f || back_measure.area == 0.0f )
      return;

    patch_node child = node;
    child.child[0] = child.child[1] = -1;

    child.area = front_measure.area;
    child.origin = front_measure.balance;
    child.normal = geometry.phong_normal( node.face, front_measure.balance );
    const uint32_t first = add_node( front, child );

    child.area = back_measure.area;
    child.origin = back_measure.balance;
    child.normal = geometry.phong_normal( node.face, back_measure.balance );
    const uint32_t second = add_node( back, child );

    tree.nodes[index].child[0] = int32_t( first );
    tree.nodes[index].child[1] = int32_t( second );

    subdivide( first, luxscale );
    subdivide( second, luxscale );
  }
};

patch_tree::patch_tree( const bsp_file& bsp, const scene_geometry& geometry )
{
  face_root.assign( geometry.faces.size(), -1 );
  root_plane.resize( geometry.faces.size() );
  builder splitter( geometry, *this );

  std::vector<uint32_t> roots;
  std::vector<float> root_luxscale;

  for ( size_t face_index = 0; face_index < geometry.faces.size(); ++face_index )
  {
    const face_info& info = geometry.faces[face_index];
    const dface& face = bsp.faces[face_index];
    if ( face.dispinfo >= 0 || info.vertex_count < 3 || face.texinfo < 0 )
      continue;

    std::vector<vec3> root_winding( geometry.face_vertex_position.begin() + info.first_vertex,
        geometry.face_vertex_position.begin() + info.first_vertex + info.vertex_count );
    remove_colinear_points( root_winding );

    if ( root_winding.size() < 3 || root_winding.size() > size_t( max_winding_points ) )
      continue;

    const float area = measure_winding( root_winding ).area;
    if ( area <= 0.0f )
      continue;

    const dtexinfo& texinfo = bsp.texinfos[face.texinfo];

    patch_node root;
    root.origin = winding_center( root_winding );
    root.normal = info.plane_normal;
    root.area = area;
    root.face = uint32_t( face_index );
    root.child[0] = root.child[1] = -1;
    root.cluster = -1;
    root.sky = ( texinfo.flags & surf::sky ) ? 1 : 0;

    face_root[face_index] = int32_t( nodes.size() );
    root_plane[face_index] = vec4( info.plane_normal, info.plane_dist );
    roots.push_back( splitter.add_node( root_winding, root ) );

    const bool whole = ( texinfo.flags & surf::nochop ) ||
                       ( ( texinfo.flags & surf::nolight ) && !( texinfo.flags & surf::light ) );
    const float lux_s = glm::length( vec3( texinfo.lightmap_vecs[0] ) );
    const float lux_t = glm::length( vec3( texinfo.lightmap_vecs[1] ) );
    root_luxscale.push_back( whole ? 0.0f : 0.5f * ( lux_s + lux_t ) );
  }

  for ( size_t r = 0; r < roots.size(); ++r )
    splitter.subdivide( roots[r], root_luxscale[r] );

  for ( patch_node& node : nodes )
  {
    node.cluster = bsp.leaf_at( node.origin ).cluster;

    for ( uint32_t j = 0; node.cluster == -1 && j < node.winding_count; ++j )
      node.cluster = bsp.leaf_at( winding[node.winding_first + j] ).cluster;
  }

  face_leaf_first_.resize( geometry.faces.size() + 1 );

  for ( uint32_t i = 0; i < nodes.size(); ++i )
  {
    if ( nodes[i].leaf() )
    {
      leaves.push_back( i );
      ++face_leaf_first_[nodes[i].face + 1];
    }
  }

  std::inclusive_scan( face_leaf_first_.begin(), face_leaf_first_.end(), face_leaf_first_.begin() );

  face_leaves_.resize( leaves.size() );
  std::vector<uint32_t> cursor( face_leaf_first_.begin(), face_leaf_first_.end() - 1 );

  for ( const uint32_t leaf : leaves )
    face_leaves_[cursor[nodes[leaf].face]++] = leaf;
}

// utils/vrad/lightmap.cpp:2060
std::vector<vec3> patch_tree::add_direct_light(
    const scene_geometry& geometry, const luxel_grid& luxels, const std::vector<vec3>& raw ) const
{
  std::vector<vec3> sample_light( nodes.size() );
  std::vector<float> sample_area( nodes.size() );

  for ( size_t face_index = 0; face_index < geometry.faces.size(); ++face_index )
  {
    const face_info& face = geometry.faces[face_index];
    if ( !face.lightmapped || face_root[face_index] < 0 )
      continue;

    const uint32_t leaf_first = face_leaf_first_[face_index];
    const uint32_t leaf_end = face_leaf_first_[face_index + 1];

    std::vector<leaf_bounds> bounds;

    for ( uint32_t l = leaf_first; l < leaf_end; ++l )
    {
      const patch_node& node = nodes[face_leaves_[l]];
      if ( node.sky )
        continue;

      leaf_bounds b{ face_leaves_[l], vec3( float_max ), vec3( float_lowest ) };

      for ( uint32_t j = 0; j < node.winding_count; ++j )
      {
        const vec3& point = winding[node.winding_first + j];
        b.mins = glm::min( b.mins, point );
        b.maxs = glm::max( b.maxs, point );
      }

      bounds.push_back( b );
    }

    for ( size_t k = 0; k < face.luxel_count(); ++k )
    {
      const uint32_t index = face.first_luxel + uint32_t( k );
      const luxel& sample = luxels.samples[index];
      if ( sample.world_area <= 0.0f )
        continue;

      const vec3& light = raw[index];

      // utils/vrad/lightmap.cpp:2078
      const float radius = glm::sqrt( sample.world_area ) / 2.0f;

      for ( const leaf_bounds& b : bounds )
      {
        bool inside = true;

        for ( int axis = 0; axis < 3 && inside; ++axis )
        {
          if ( b.mins[axis] > sample.position[axis] + radius ||
               b.maxs[axis] < sample.position[axis] - radius )
            inside = false;
        }

        if ( !inside )
          continue;

        sample_area[b.node] += sample.world_area;
        sample_light[b.node] += light * sample.world_area;
      }
    }
  }

  std::vector<vec3> total( nodes.size() );

  for ( size_t i = 0; i < nodes.size(); ++i )
  {
    if ( sample_area[i] > 0.0f )
      total[i] = sample_light[i] / sample_area[i];
  }

  // utils/vrad/lightmap.cpp:3270
  for ( size_t i = nodes.size(); i-- > 0; )
  {
    const patch_node& node = nodes[i];
    if ( node.leaf() )
      continue;

    const patch_node& a = nodes[node.child[0]];
    const patch_node& b = nodes[node.child[1]];
    const float sum = a.area + b.area;
    total[i] = total[node.child[0]] * ( a.area / sum ) + total[node.child[1]] * ( b.area / sum );
  }

  return total;
}

// utils/vrad/vrad.cpp:1413
vec3 patch_tree::collect_light(
    const std::vector<vec3>& received, std::vector<vec3>& emit, std::vector<vec3>& total ) const
{
  vec3 added( 0.0f );

  for ( size_t i = nodes.size(); i-- > 0; )
  {
    const patch_node& node = nodes[i];
    if ( node.sky )
    {
      emit[i] = vec3( 0.0f );
    }
    else if ( node.leaf() )
    {
      total[i] += received[i];
      emit[i] = received[i];
      added += emit[i];
    }
    else
    {
      const patch_node& a = nodes[node.child[0]];
      const patch_node& b = nodes[node.child[1]];
      const float sum = a.area + b.area;
      emit[i] = emit[node.child[0]] * ( a.area / sum ) + emit[node.child[1]] * ( b.area / sum );
    }
  }

  return added;
}

namespace
{

void splat_leaf( const patch_tree& tree, const face_info& face, uint32_t leaf, const vec3& light,
    std::vector<vec3>& value, std::vector<float>& weight )
{
  const patch_node& node = tree.nodes[leaf];
  const vec2 coord = world_to_st( face.world_to_luxel_s, face.world_to_luxel_t, node.origin );

  // utils/vrad/radial.cpp:237
  vec2 mins( float_max );
  vec2 maxs( float_lowest );

  for ( uint32_t j = 0; j < node.winding_count; ++j )
  {
    const vec2 c = world_to_st(
        face.world_to_luxel_s, face.world_to_luxel_t, tree.winding[node.winding_first + j] );
    mins = glm::min( mins, c );
    maxs = glm::max( maxs, c );
  }

  const float dists = glm::max( 1.0f, maxs.x - mins.x );
  const float distt = glm::max( 1.0f, maxs.y - mins.y );

  const int32_t s_min = glm::max( int32_t( coord.x - dists * radial_dist ), 0 );
  const int32_t t_min = glm::max( int32_t( coord.y - distt * radial_dist ), 0 );
  const int32_t s_max = glm::min( int32_t( coord.x + dists * radial_dist + 1.0f ), face.width );
  const int32_t t_max = glm::min( int32_t( coord.y + distt * radial_dist + 1.0f ), face.height );

  for ( int32_t s = s_min; s < s_max; ++s )
  {
    for ( int32_t t = t_min; t < t_max; ++t )
    {
      const float ds = ( coord.x - float( s ) ) / dists;
      const float dt = ( coord.y - float( t ) ) / distt;
      const float r = radial_dist2 - ( ds * ds + dt * dt );
      if ( r > 0.0f )
      {
        const size_t i = size_t( t ) * size_t( face.width ) + size_t( s );
        value[i] += light * r;
        weight[i] += r;
      }
    }
  }
}

} // namespace

// utils/vrad/radial.cpp:157
void patch_tree::splat_bounce( const scene_geometry& geometry, const std::vector<vec3>& bounce,
    std::vector<vec3>& delta, progress& bounce_progress ) const
{
  const size_t lit_faces =
      size_t( std::ranges::count( geometry.faces, true, &face_info::lightmapped ) );
  std::atomic<size_t> splatted = 0;

  tbb::parallel_for( size_t( 0 ), geometry.faces.size(),
      [&]( size_t face_index )
      {
        const face_info& face = geometry.faces[face_index];
        if ( !face.lightmapped )
          return;

        std::vector<vec3> value( face.luxel_count() );
        std::vector<float> weight( value.size() );

        const auto splat_face = [&]( uint32_t source_face )
        {
          for ( uint32_t l = face_leaf_first_[source_face]; l < face_leaf_first_[source_face + 1];
              ++l )
          {
            const uint32_t leaf = face_leaves_[l];
            splat_leaf( *this, face, leaf, bounce[leaf], value, weight );
          }
        };

        splat_face( uint32_t( face_index ) );

        for ( uint32_t n = geometry.face_neighbor_first[face_index];
            n < geometry.face_neighbor_first[face_index + 1]; ++n )
        {
          splat_face( geometry.face_neighbors[n] );
        }

        for ( size_t k = 0; k < value.size(); ++k )
        {
          const uint32_t index = face.first_luxel + uint32_t( k );
          delta[index] = weight[k] > radial_weight_eps ? value[k] / weight[k] : vec3( 0.0f );
        }

        bounce_progress.set( ++splatted, lit_faces );
      } );
}
