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

struct splat_grid
{
  splat_grid( int32_t w, int32_t h )
      : width( w ), height( h ), value( size_t( w ) * size_t( h ) ), weight( value.size() )
  {
  }

  void add( int32_t s, int32_t t, const vec3& light, float r )
  {
    const size_t i = size_t( t ) * size_t( width ) + size_t( s );
    value[i] += light * r;
    weight[i] += r;
  }

  vec3 resolve( size_t i ) const
  {
    return weight[i] > radial_weight_eps ? value[i] / weight[i] : vec3( 0.0f );
  }

  int32_t width;
  int32_t height;
  std::vector<vec3> value;
  std::vector<float> weight;
};

std::vector<uint32_t> steep_gradient_luxels( const scene_geometry& geometry,
    const luxel_grid& luxels, const std::vector<vec3>& delta, std::vector<uint8_t>& processed );

void radial_filter(
    const scene_geometry& geometry, const luxel_grid& luxels, std::vector<vec3>& delta );
