#pragma once

#include "bsp.h"
#include "geometry.h"
#include "log.h"
#include "luxels.h"
#include "math.h"

#include <vector>

struct patch_node
{
  vec3 origin;
  vec3 normal;
  float area;
  uint32_t face;
  int32_t child[2];
  uint32_t winding_first;
  uint32_t winding_count;
  int32_t cluster;
  uint8_t sky;

  bool leaf() const
  {
    return child[0] < 0;
  }
};

class patch_tree
{
public:
  patch_tree() = default;
  patch_tree( const bsp_file& bsp, const scene_geometry& geometry );

  std::vector<vec3> add_direct_light( const scene_geometry& geometry, const luxel_grid& luxels,
      const std::vector<vec3>& raw ) const;

  vec3 collect_light(
      const std::vector<vec3>& received, std::vector<vec3>& emit, std::vector<vec3>& total ) const;

  void splat_bounce( const scene_geometry& geometry, const std::vector<vec3>& bounce,
      std::vector<vec3>& delta, progress& bounce_progress ) const;

  std::vector<patch_node> nodes;
  std::vector<vec3> winding;
  std::vector<vec2> winding_uv; // for displacements
  std::vector<int32_t> face_root;
  std::vector<vec4> root_plane;
  std::vector<uint32_t> leaves;

private:
  struct builder;

  std::vector<uint32_t> face_leaf_first_;
  std::vector<uint32_t> face_leaves_;
};
