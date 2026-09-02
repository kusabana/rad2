#include "bsp.h"

#include "geometry.h"
#include "lights.h"

#include <algorithm>
#include <bit>
#include <cstring>
#include <fstream>
#include <limits>

namespace
{

namespace lump
{
enum : int32_t
{
  entities = 0,
  planes = 1,
  texdata = 2,
  vertexes = 3,
  visibility = 4,
  nodes = 5,
  texinfo = 6,
  faces = 7,
  lighting = 8,
  leafs = 10,
  edges = 12,
  surfedges = 13,
  models = 14,
  worldlights = 15,
  leaffaces = 16,
  leafbrushes = 17,
  brushes = 18,
  brushsides = 19,
  dispinfo = 26,
  vertnormals = 30,
  vertnormalindices = 31,
  disp_verts = 33,
  game_lump = 35,
  pakfile = 40,
  lighting_hdr = 53,
  worldlights_hdr = 54,
  faces_hdr = 58,
};
}

// public/bspfile.h:21
constexpr int32_t bsp_ident = ( 'P' << 24 ) + ( 'S' << 16 ) + ( 'B' << 8 ) + 'V';
// public/bspfile.h:24
constexpr int32_t bsp_version_min = 19;
constexpr int32_t bsp_version_max = 20;

// public/bspfile.h:93
constexpr int32_t max_map_worldlights = 8192;
// public/mathlib/bumpvects.h:25
constexpr int32_t num_bump_vects = 3;

constexpr uint8_t no_lightstyle = std::numeric_limits<uint8_t>::max();

constexpr int float_exponent_bias = std::numeric_limits<float>::max_exponent - 1;
constexpr int rgb_exp32_mantissa_shift = std::numeric_limits<uint8_t>::digits - 1;
constexpr int exponent_min = std::numeric_limits<int8_t>::min();

// utils/vrad/vismat.cpp:33
constexpr float test_epsilon = 0.1f;

// mathlib/color_conversion.cpp:566
color_rgb_exp32 encode_rgb_exp32( const vec3& color )
{
  const float max_value = max_component( color );
  if ( !( max_value > 0.0f ) )
    return { 0, 0, 0, 0 };

  const int float_exponent =
      int( std::bit_cast<uint32_t>( max_value ) >> float_mantissa_bits ) - float_exponent_bias;
  int exponent = glm::max( float_exponent - rgb_exp32_mantissa_shift, exponent_min );
  if ( max_value * glm::ldexp( 1.0f, -exponent ) + 0.5f >= byte_range )
    ++exponent;

  const vec3 scaled = color * glm::ldexp( 1.0f, -exponent ) + 0.5f;
  return { uint8_t( scaled.x ), uint8_t( scaled.y ), uint8_t( scaled.z ), int8_t( exponent ) };
}

// utils/vrad/radial.cpp:855
vec3 face_median( std::span<const vec3> values )
{
  if ( values.empty() )
    return vec3( 0.0f );

  std::vector<float> channel( values.size() );
  vec3 median( 0.0f );

  for ( int c = 0; c < 3; ++c )
  {
    for ( size_t i = 0; i < values.size(); ++i )
      channel[i] = values[i][c];

    const size_t mid = channel.size() / 2;
    std::nth_element( channel.begin(), channel.begin() + mid, channel.end() );
    median[c] = channel[mid];
  }

  return median;
}

std::vector<std::byte> read_file_bytes( const std::filesystem::path& path )
{
  std::ifstream stream( path, std::ios::binary | std::ios::ate );
  if ( !stream )
    throw fatal_error( std::format( "cannot open '{}'", path.string() ) );

  const std::streamoff size = stream.tellg();
  stream.seekg( 0 );

  std::vector<std::byte> bytes( static_cast<size_t>( size ) );
  if ( size > 0 && !stream.read( reinterpret_cast<char*>( bytes.data() ), size ) )
    throw fatal_error( std::format( "failed reading '{}'", path.string() ) );

  return bytes;
}

void write_file_bytes( const std::filesystem::path& path, std::span<const std::byte> bytes )
{
  std::ofstream stream( path, std::ios::binary | std::ios::trunc );
  if ( !stream )
    throw fatal_error( std::format( "cannot create '{}'", path.string() ) );

  if ( !stream.write(
           reinterpret_cast<const char*>( bytes.data() ), std::streamsize( bytes.size() ) ) )
  {
    throw fatal_error( std::format( "failed writing '{}'", path.string() ) );
  }
}

} // namespace

std::span<const std::byte> bsp_file::lump_bytes( int32_t lump_id ) const
{
  if ( replaced_[lump_id] )
    return replaced_[lump_id]->bytes;

  const dlump& entry = header_.lumps[lump_id];
  if ( entry.file_len <= 0 )
    return {};

  return { file_bytes_.data() + entry.file_ofs, size_t( entry.file_len ) };
}

template <typename element_t> std::span<const element_t> bsp_file::lump( int32_t lump_id ) const
{
  const std::span<const std::byte> bytes = lump_bytes( lump_id );
  if ( bytes.size() % sizeof( element_t ) != 0 )
  {
    throw fatal_error( std::format( "lump {} size {} is not a multiple of element size {}", lump_id,
        bytes.size(), sizeof( element_t ) ) );
  }

  return { reinterpret_cast<const element_t*>( bytes.data() ), bytes.size() / sizeof( element_t ) };
}

template <typename element_t>
void bsp_file::set_lump( int32_t lump_id, std::span<const element_t> data, int32_t version )
{
  const std::byte* begin = reinterpret_cast<const std::byte*>( data.data() );
  replaced_[lump_id] =
      replaced_lump{ std::vector<std::byte>( begin, begin + data.size_bytes() ), version };
}

bsp_file::bsp_file( const std::filesystem::path& path ) : file_bytes_( read_file_bytes( path ) )
{
  if ( file_bytes_.size() < sizeof( dheader ) )
    throw fatal_error( std::format( "'{}' is too small to be a bsp", path.string() ) );

  std::memcpy( &header_, file_bytes_.data(), sizeof( dheader ) );

  if ( header_.ident != bsp_ident )
    throw fatal_error( std::format( "'{}' uses an unsupported BSP format", path.string() ) );

  if ( header_.version < bsp_version_min || header_.version > bsp_version_max )
    throw fatal_error( std::format( "'{}' uses an unsupported BSP version", path.string() ) );

  for ( int32_t id = 0; id < bsp_header_lumps; ++id )
  {
    const dlump& entry = header_.lumps[id];
    if ( entry.file_len == 0 )
      continue;

    if ( entry.file_len < 0 || entry.file_ofs < 0 ||
         int64_t( entry.file_ofs ) + entry.file_len > int64_t( file_bytes_.size() ) )
      throw fatal_error( std::format( "lump {} is out of bounds", id ) );

    if ( std::bit_cast<uint32_t>( entry.four_cc ) != 0 )
      throw fatal_error( std::format( "lump {} is compressed", id ) );
  }

  const std::span<const std::byte> entities = lump_bytes( lump::entities );
  entity_text = { reinterpret_cast<const char*>( entities.data() ), entities.size() };
  visibility = lump_bytes( lump::visibility );
  planes = lump<dplane>( lump::planes );
  vertices = lump<dvertex>( lump::vertexes );
  edges = lump<dedge>( lump::edges );
  surfedges = lump<int32_t>( lump::surfedges );
  faces = lump<dface>( lump::faces );
  texinfos = lump<dtexinfo>( lump::texinfo );
  texdatas = lump<dtexdata>( lump::texdata );
  models = lump<dmodel>( lump::models );
  nodes = lump<dnode>( lump::nodes );
  brushes = lump<dbrush>( lump::brushes );
  brushsides = lump<dbrushside>( lump::brushsides );
  leaf_brushes = lump<uint16_t>( lump::leafbrushes );
  leaf_faces = lump<uint16_t>( lump::leaffaces );
  dispinfos = lump<ddispinfo>( lump::dispinfo );
  disp_verts = lump<ddispvert>( lump::disp_verts );

  if ( header_.lumps[lump::leafs].version == 0 )
  {
    const std::span<const dleaf_v0> v0 = lump<dleaf_v0>( lump::leafs );
    leaves.assign( v0.begin(), v0.end() );
  }
  else
  {
    const std::span<const dleaf_v1> v1 = lump<dleaf_v1>( lump::leafs );
    leaves.assign( v1.begin(), v1.end() );
  }
}

// utils/vrad/vismat.cpp:95
const dleaf_v1& bsp_file::leaf_at( const vec3& point, int32_t node_index ) const
{
  while ( node_index >= 0 )
  {
    const dnode& node = nodes[node_index];
    const dplane& plane = planes[node.plane_num];
    const float dist = glm::dot( point, plane.normal ) - plane.dist;
    if ( dist > test_epsilon )
    {
      node_index = node.children[0];
    }
    else if ( dist < -test_epsilon )
    {
      node_index = node.children[1];
    }
    else
    {
      const dleaf_v1& front = leaf_at( point, node.children[0] );
      if ( front.cluster != -1 )
        return front;

      return leaf_at( point, node.children[1] );
    }
  }

  return leaves[size_t( -node_index - 1 )];
}

void bsp_file::write_lighting(
    const scene_geometry& geometry, const std::vector<vec3>& lightmap, lighting_target target )
{
  std::vector<dface> updated( faces.begin(), faces.end() );
  std::vector<color_rgb_exp32> samples;

  for ( size_t face_index = 0; face_index < geometry.faces.size(); ++face_index )
  {
    const face_info& info = geometry.faces[face_index];
    dface& face = updated[face_index];
    std::ranges::fill( face.styles, no_lightstyle );

    if ( !info.lightmapped )
    {
      face.light_ofs = -1;
      continue;
    }

    const std::span<const vec3> face_values(
        lightmap.data() + info.first_luxel, info.luxel_count() );
    face.styles[0] = 0;

    // utils/vrad/lightmap.cpp:3371
    samples.push_back( encode_rgb_exp32( face_median( face_values ) ) );

    const size_t offset = samples.size() * sizeof( color_rgb_exp32 );
    if ( offset > size_t( std::numeric_limits<int32_t>::max() ) )
      throw fatal_error( "lighting lump exceeds 2 GiB" );

    face.light_ofs = int32_t( offset );

    const int bump_layers = ( info.surf_flags & surf::bumplight ) ? 1 + num_bump_vects : 1;
    for ( int layer = 0; layer < bump_layers; ++layer )
    {
      for ( const vec3& value : face_values )
        samples.push_back( encode_rgb_exp32( value ) );
    }
  }

  if ( target == lighting_target::ldr )
  {
    set_lump<color_rgb_exp32>( lump::lighting, samples, 1 );
    set_lump<dface>( lump::faces, updated, header_.lumps[lump::faces].version );
  }
  else
  {
    set_lump<color_rgb_exp32>( lump::lighting_hdr, samples, 1 );
    set_lump<dface>( lump::faces_hdr, updated, 1 );
  }
}

void bsp_file::write_worldlights( const light_table& table, lighting_target target )
{
  std::vector<dworldlight> worldlights;

  for ( const light& source : table.lights )
  {
    dworldlight wl = source;
    wl.intensity = source.intensity / byte_max;
    wl.cluster = leaf_at( source.origin ).cluster;
    worldlights.push_back( wl );
  }

  if ( worldlights.size() > size_t( max_map_worldlights ) )
  {
    throw fatal_error( std::format(
        "{} worldlights exceed the engine cap of {}", worldlights.size(), max_map_worldlights ) );
  }

  set_lump<dworldlight>(
      target == lighting_target::hdr ? lump::worldlights_hdr : lump::worldlights, worldlights, 0 );
}

void bsp_file::write_vertnormals( const scene_geometry& geometry )
{
  set_lump<vec3>( lump::vertnormals, geometry.vertex_normal_table, 0 );
  set_lump<uint16_t>( lump::vertnormalindices, geometry.vertex_normal_indices, 0 );
}

std::vector<std::byte> bsp_file::rebuild_file_bytes() const
{
  struct placement
  {
    int32_t id;
    int64_t original_ofs;
  };

  std::vector<placement> order;
  std::vector<placement> fresh;

  int32_t last_original_id = -1;
  int64_t last_original_ofs = -1;

  for ( int32_t id = 0; id < bsp_header_lumps; ++id )
  {
    const dlump& entry = header_.lumps[id];
    const bool originally_present = entry.file_len > 0;
    const bool now_present = replaced_[id] ? !replaced_[id]->bytes.empty() : originally_present;

    if ( originally_present && entry.file_ofs > last_original_ofs )
    {
      last_original_ofs = entry.file_ofs;
      last_original_id = id;
    }

    if ( !now_present )
      continue;

    if ( originally_present )
      order.push_back( { .id = id, .original_ofs = entry.file_ofs } );
    else
      fresh.push_back( { .id = id, .original_ofs = 0 } );
  }

  std::ranges::sort( order, {}, &placement::original_ofs );

  if ( !fresh.empty() )
  {
    const auto insert_at = last_original_id == lump::pakfile
                               ? std::ranges::find( order, lump::pakfile, &placement::id )
                               : order.end();
    order.insert( insert_at, fresh.begin(), fresh.end() );
  }

  dheader new_header = header_;

  for ( int32_t id = 0; id < bsp_header_lumps; ++id )
  {
    new_header.lumps[id].file_ofs = 0;
    new_header.lumps[id].file_len = 0;
    new_header.lumps[id].version =
        replaced_[id] ? replaced_[id]->version : header_.lumps[id].version;
  }

  std::vector<std::byte> output;
  output.resize( sizeof( dheader ) );

  const int64_t old_gamelump_ofs = header_.lumps[lump::game_lump].file_ofs;
  int64_t new_gamelump_ofs = -1;

  for ( const placement& item : order )
  {
    // every lump starts four byte aligned
    output.resize( ( output.size() + 3 ) & ~size_t( 3 ) );

    const std::span<const std::byte> data = lump_bytes( item.id );

    if ( output.size() + data.size() > size_t( std::numeric_limits<int32_t>::max() ) )
      throw fatal_error( "output bsp would exceed 2 GiB" );

    new_header.lumps[item.id].file_ofs = int32_t( output.size() );
    new_header.lumps[item.id].file_len = int32_t( data.size() );

    if ( item.id == lump::game_lump )
      new_gamelump_ofs = int64_t( output.size() );

    output.insert( output.end(), data.begin(), data.end() );
  }

  if ( new_gamelump_ofs >= 0 && !replaced_[lump::game_lump] &&
       new_gamelump_ofs != old_gamelump_ofs &&
       new_header.lumps[lump::game_lump].file_len >= int32_t( sizeof( dgamelump_header ) ) )
  {
    const int64_t delta = new_gamelump_ofs - old_gamelump_ofs;

    dgamelump_header game_header = {};
    std::memcpy( &game_header, output.data() + new_gamelump_ofs, sizeof( game_header ) );

    for ( int32_t i = 0; i < game_header.lump_count; ++i )
    {
      std::byte* entry_bytes = output.data() + new_gamelump_ofs + sizeof( dgamelump_header ) +
                               size_t( i ) * sizeof( dgamelump );

      dgamelump entry = {};
      std::memcpy( &entry, entry_bytes, sizeof( entry ) );

      if ( entry.file_ofs != 0 )
      {
        entry.file_ofs = int32_t( entry.file_ofs + delta );
        std::memcpy( entry_bytes, &entry, sizeof( entry ) );
      }
    }
  }

  std::memcpy( output.data(), &new_header, sizeof( new_header ) );
  return output;
}

void bsp_file::save( const std::filesystem::path& path ) const
{
  const bool any_replaced =
      std::ranges::any_of( replaced_, []( const auto& slot ) { return slot.has_value(); } );

  write_file_bytes( path, any_replaced ? rebuild_file_bytes() : file_bytes_ );
}
