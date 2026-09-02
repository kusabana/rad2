#pragma once

#include "log.h"
#include "math.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

class scene_geometry;
class light_table;
enum class lighting_target : uint8_t;

// public/bspfile.h:370
constexpr int32_t bsp_header_lumps = 64;

// texinfo surface flags
// public/bspflags.h:80
namespace surf
{
enum : int32_t
{
  light = 0x0001,
  sky2d = 0x0002,
  sky = 0x0004,
  nodraw = 0x0080,
  hint = 0x0100,
  skip = 0x0200,
  nolight = 0x0400,
  bumplight = 0x0800,
  nochop = 0x4000,
};
}

// brush contents
// public/bspflags.h:26
namespace contents
{
enum : uint32_t
{
  solid = 0x1,
  opaque = 0x80,
  moveable = 0x4000,
};

// public/bspflags.h:114
constexpr uint32_t mask_opaque = solid | moveable | opaque;
} // namespace contents

// public/bspfile.h:953
enum class emit_type : int32_t
{
  point = 1,
  spotlight = 2,
  skylight = 3,
  skyambient = 5,
};

#pragma pack( push, 1 )

struct dlump
{
  int32_t file_ofs;
  int32_t file_len;
  int32_t version;
  char four_cc[4];
};

struct dheader
{
  int32_t ident;
  int32_t version;
  dlump lumps[bsp_header_lumps];
  int32_t map_revision;
};

struct dplane
{
  vec3 normal;
  float dist;
  int32_t type;
};

struct dvertex
{
  vec3 position;
};

struct dedge
{
  uint16_t vertex[2];
};

struct dface
{
  uint16_t plane_num;
  uint8_t side;
  uint8_t on_node;
  int32_t first_edge;
  int16_t num_edges;
  int16_t texinfo;
  int16_t dispinfo;
  int16_t surface_fog_volume_id;
  uint8_t styles[4];
  int32_t light_ofs;
  float area;
  int32_t lightmap_mins[2];
  int32_t lightmap_size[2];
  int32_t orig_face;
  uint16_t num_prims;
  uint16_t first_prim_id;
  uint32_t smoothing_groups;
};

struct dtexinfo
{
  vec4 texture_vecs[2];
  vec4 lightmap_vecs[2];
  int32_t flags;
  int32_t texdata;
};

struct dtexdata
{
  vec3 reflectivity;
  int32_t name_string_table_id;
  int32_t width;
  int32_t height;
  int32_t view_width;
  int32_t view_height;
};

struct dmodel
{
  vec3 mins;
  vec3 maxs;
  vec3 origin;
  int32_t head_node;
  int32_t first_face;
  int32_t num_faces;
};

struct dnode
{
  int32_t plane_num;
  int32_t children[2];
  int16_t mins[3];
  int16_t maxs[3];
  uint16_t first_face;
  uint16_t num_faces;
  int16_t area;
  int16_t padding;
};

// nine bits of area in the leaf flags
// public/bspfile.h:833
constexpr int16_t leaf_area_mask = 0x1FF;

struct dleaf_v1
{
  int32_t contents;
  int16_t cluster;
  int16_t area_flags;
  int16_t mins[3];
  int16_t maxs[3];
  uint16_t first_leaf_face;
  uint16_t num_leaf_faces;
  uint16_t first_leaf_brush;
  uint16_t num_leaf_brushes;
  int16_t leaf_water_data_id;
  int16_t padding;

  int32_t area() const
  {
    return int32_t( area_flags & leaf_area_mask );
  }
};

struct dbrush
{
  int32_t first_side;
  int32_t num_sides;
  int32_t contents;
};

struct dbrushside
{
  uint16_t plane_num;
  int16_t texinfo;
  int16_t dispinfo;
  int16_t bevel;
};

struct dworldlight
{
  vec3 origin;
  vec3 intensity;
  vec3 normal;
  int32_t cluster;
  emit_type type;
  int32_t style;
  float stopdot;
  float stopdot2;
  float exponent;
  float radius;
  float constant_attn;
  float linear_attn;
  float quadratic_attn;
  int32_t flags;
  int32_t texinfo;
  int32_t owner;

  bool operator==( const dworldlight& ) const = default;
};

struct ddispinfo
{
  vec3 start_position;
  int32_t disp_vert_start;
  int32_t disp_tri_start;
  int32_t power;
  int32_t min_tess;
  float smoothing_angle;
  int32_t contents;
  uint16_t map_face;
  uint16_t padding0;
  int32_t lightmap_alpha_start;
  int32_t lightmap_sample_position_start;
  uint8_t neighbours[128];
};

struct ddispvert
{
  vec3 vec;
  float dist;
  float alpha;
};

struct dgamelump_header
{
  int32_t lump_count;
};

struct dgamelump
{
  int32_t id;
  uint16_t flags;
  uint16_t version;
  int32_t file_ofs;
  int32_t file_len;
};

struct color_rgb_exp32
{
  uint8_t r;
  uint8_t g;
  uint8_t b;
  int8_t exponent;
};

struct compressed_light_cube
{
  color_rgb_exp32 color[6];
};

struct dleaf_v0 : dleaf_v1
{
  compressed_light_cube ambient;
};

#pragma pack( pop )

class bsp_file
{
public:
  explicit bsp_file( const std::filesystem::path& path );

  bsp_file( const bsp_file& ) = delete;
  bsp_file& operator=( const bsp_file& ) = delete;

  int32_t version() const
  {
    return header_.version;
  }

  int32_t map_revision() const
  {
    return header_.map_revision;
  }

  // utils/vrad/vismat.cpp:95
  const dleaf_v1& leaf_at( const vec3& point, int32_t node_index = 0 ) const;

  void write_lighting(
      const scene_geometry& geometry, const std::vector<vec3>& lightmap, lighting_target target );

  void write_worldlights( const light_table& table, lighting_target target );

  void write_vertnormals( const scene_geometry& geometry );

  void save( const std::filesystem::path& path ) const;

  std::string_view entity_text;
  std::span<const std::byte> visibility;
  std::span<const dplane> planes;
  std::span<const dvertex> vertices;
  std::span<const dedge> edges;
  std::span<const int32_t> surfedges;
  std::span<const dface> faces;
  std::span<const dtexinfo> texinfos;
  std::span<const dtexdata> texdatas;
  std::span<const dmodel> models;
  std::span<const dnode> nodes;
  std::vector<dleaf_v1> leaves;
  std::span<const dbrush> brushes;
  std::span<const dbrushside> brushsides;
  std::span<const uint16_t> leaf_brushes;
  std::span<const uint16_t> leaf_faces;
  std::span<const ddispinfo> dispinfos;
  std::span<const ddispvert> disp_verts;

private:
  struct replaced_lump
  {
    std::vector<std::byte> bytes;
    int32_t version;
  };

  std::span<const std::byte> lump_bytes( int32_t lump_id ) const;
  template <typename element_t> std::span<const element_t> lump( int32_t lump_id ) const;
  template <typename element_t>
  void set_lump( int32_t lump_id, std::span<const element_t> data, int32_t version );
  std::vector<std::byte> rebuild_file_bytes() const;

  std::vector<std::byte> file_bytes_;
  dheader header_ = {};
  std::array<std::optional<replaced_lump>, bsp_header_lumps> replaced_;
};
