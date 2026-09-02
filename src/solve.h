#pragma once

#include "geometry.h"
#include "lights.h"
#include "log.h"
#include "luxels.h"
#include "math.h"
#include "patches.h"
#include "visibility.h"

#include <span>
#include <string>
#include <string_view>
#include <vector>

// utils/vrad/vrad.cpp:51
constexpr uint32_t vrad_bounce_cap = 100;

// utils/vrad/lightmap.cpp:2694
constexpr uint32_t extra_grid = 4;
constexpr uint32_t sky_extra_grid = 2;

// utils/vrad/lightmap.cpp:1669
constexpr uint32_t vrad_sun_samples = 30;

// public/mathlib/anorms.h:17
constexpr uint32_t vrad_sky_directions = 162;
// utils/vrad/vrad.cpp:2500
constexpr uint32_t final_sky_scale = 16;

struct solve_settings
{
  uint32_t bounce_cap = vrad_bounce_cap;
  uint32_t position_samples = extra_grid * extra_grid;
  uint32_t sky_position_samples = sky_extra_grid * sky_extra_grid;
  uint32_t sun_samples = vrad_sun_samples;
  uint32_t sky_samples = vrad_sky_directions * final_sky_scale;
};

struct compute_device_info
{
  uint32_t index = 0;
  std::string name;
  std::string vendor;
  bool embree_supported = false;
};

// all SYCL GPU devices visible on the system
std::vector<compute_device_info> list_devices();

struct direct_layers
{
  std::string_view name;
  std::vector<vec3> lightmap;
  std::vector<vec3> raw;
};

direct_layers direct_light( const scene_geometry& geometry, const luxel_grid& luxels,
    const light_table& table, const solve_settings& settings, uint32_t device );

std::vector<std::vector<vec3>> bounce_light( const scene_geometry& geometry,
    const luxel_grid& luxels, const patch_tree& tree, const transfer_tables& tables,
    std::span<const direct_layers> direct, uint32_t bounce_cap, uint32_t device );
