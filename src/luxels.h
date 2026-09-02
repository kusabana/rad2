#pragma once

#include "bsp.h"
#include "geometry.h"
#include "math.h"

#include <vector>

struct luxel
{
  vec3 position;
  uint32_t face_index;
  vec3 normal;
  float world_area;
  vec2 luxel_position; // luxel space
  int32_t area;
};

// samples sit one unit off their face
// utils/vrad/lightmap.cpp:2451
constexpr float sample_lift = 1.0f;

class luxel_grid
{
public:
  luxel_grid( const bsp_file& bsp, const scene_geometry& geometry );

  std::vector<luxel> samples;
  std::vector<vec2> cell_mins;
  std::vector<vec2> cell_maxs;

private:
  void find_leaves( const scene_geometry& geometry, const bsp_file& bsp );
};
