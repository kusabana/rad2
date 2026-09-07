#pragma once

#include "bsp.h"
#include "entities.h"
#include "math.h"

#include <vector>

struct edge_plane
{
  vec2 inward;
  float dist;
};

struct face_info
{
  bool lightmapped = false;
  int32_t surf_flags = 0;

  mat3 luxel_to_world = mat3( 0.0f );
  vec4 world_to_luxel_s = vec4( 0.0f );
  vec4 world_to_luxel_t = vec4( 0.0f );

  vec3 plane_normal = vec3( 0.0f, 0.0f, 1.0f );
  float plane_dist = 0.0f;

  vec3 reflectivity = vec3( 0.0f );
  int32_t width = 0;
  int32_t height = 0;

  int32_t model_index = 0;
  vec3 model_origin = vec3( 0.0f );
  int32_t orig_face = -1;

  uint32_t first_vertex = 0;
  uint32_t vertex_count = 0;

  uint32_t first_luxel = 0;
  uint32_t edge_first = 0;
  uint32_t edge_count = 0;

  int32_t disp_first_vertex = -1;
  int32_t disp_power = 0;

  size_t luxel_count() const
  {
    return size_t( width ) * size_t( height );
  }

  bool displacement() const
  {
    return disp_power > 0;
  }
};

class scene_geometry
{
public:
  scene_geometry( const bsp_file& bsp, const entity_lump& entities );

  vec3 phong_normal( uint32_t face_index, const vec3& spot ) const;

  std::vector<vec3> vertices;
  std::vector<uint32_t> sky_indices;
  std::vector<uint32_t> occluder_indices;
  std::vector<face_info> faces;

  uint32_t luxel_count = 0;
  std::vector<edge_plane> edge_planes;
  std::vector<vec3> face_vertex_position;
  std::vector<uint32_t> face_neighbors;
  std::vector<uint32_t> face_neighbor_first;
  std::vector<vec3> vertex_normal_table;
  std::vector<uint16_t> vertex_normal_indices;

private:
  void build_face_infos( const bsp_file& bsp, const std::vector<int32_t>& face_model,
      const std::vector<vec3>& model_origin );
  void build_edge_planes();
  void emit_sky_triangles();
  void collect_displacement_occluders( const bsp_file& bsp );
  void build_smoothed_normals( const bsp_file& bsp );
  void dedup_vertex_normals();

  std::vector<uint32_t> face_vertex_index_;
  std::vector<vec3> face_vertex_normal_;
  std::vector<vec3> face_centroid_;
};

// utils/vrad/vrad.cpp:105
// constexpr float smooth_cos_threshold = const_cos( const_radians( 45.0f ) );
constexpr float smooth_cos_threshold = 0.7071067f;
