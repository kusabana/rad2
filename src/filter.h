#pragma once

#include "geometry.h"
#include "luxels.h"
#include "math.h"

#include <vector>

// utils/vrad/radial.h:22
constexpr float radial_dist2 = 2.0f;
// constexpr float radial_dist = const_sqrt( radial_dist2 );
constexpr float radial_dist = 1.42f;

// utils/vrad/radial.h:25
constexpr float radial_weight_eps = 1.0e-5f;

std::vector<uint32_t> steep_gradient_luxels( const scene_geometry& geometry,
    const luxel_grid& luxels, const std::vector<vec3>& delta, std::vector<uint8_t>& processed );

void radial_filter(
    const scene_geometry& geometry, const luxel_grid& luxels, std::vector<vec3>& delta );
