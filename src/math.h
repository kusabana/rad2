#pragma once

#define GLM_ENABLE_EXPERIMENTAL
#include <glm/glm.hpp>
#include <glm/gtc/constants.hpp>
#include <glm/gtx/norm.hpp>

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <vector>

using glm::mat3;
using glm::vec2;
using glm::vec3;
using glm::vec4;

inline float max_component( const vec3& v )
{
  return glm::max( v.x, glm::max( v.y, v.z ) );
}

// public/mathlib/vector.h:2210
inline vec3 safe_normalize( const vec3& v )
{
  const float length2 = glm::dot( v, v );
  return length2 > 0.0f ? v * glm::inversesqrt( length2 ) : vec3( 0.0f );
}

// public/mathlib/mathlib.h:313
constexpr float equal_epsilon = 0.001f;

// utils/vrad/lightmap.cpp:1088, :2850
constexpr double light_gamma = 2.2;

constexpr float float_max = std::numeric_limits<float>::max();
constexpr float float_lowest = std::numeric_limits<float>::lowest();

constexpr int float_mantissa_bits = std::numeric_limits<float>::digits - 1;

// eight bit colour channels
constexpr float byte_max = float( std::numeric_limits<uint8_t>::max() );
constexpr float byte_range = byte_max + 1.0f;

// public/worldsize.h:19
constexpr float max_coord_integer = 16384.0f;
constexpr float coord_extent = 2.0f * max_coord_integer;

// utils/common/polylib.h:37
constexpr float on_epsilon = 0.1f;

constexpr float length_epsilon = 1.0e-6f;
constexpr float basis_epsilon = 1.0e-20f;

// luxel coordinates of a world position
inline vec2 world_to_st( const vec4& row_s, const vec4& row_t, const vec3& position )
{
  return vec2( glm::dot( position, vec3( row_s ) ) + row_s.w,
      glm::dot( position, vec3( row_t ) ) + row_t.w );
}

inline float signed_area2( std::span<const vec2> points )
{
  float area2 = 0.0f;

  for ( size_t i = 0; i < points.size(); ++i )
  {
    const vec2& a = points[i];
    const vec2& b = points[( i + 1 ) % points.size()];
    area2 += a.x * b.y - b.x * a.y;
  }

  return area2;
}

inline std::optional<vec2> inward_edge_normal( const vec2& a, const vec2& b, float orient )
{
  const vec2 edge = b - a;
  const float length = glm::length( edge );
  if ( length < length_epsilon )
    return std::nullopt;

  return vec2( -edge.y, edge.x ) * ( orient / length );
}

template <typename vec_t>
void clip_convex( const std::vector<vec_t>& input, const vec_t& normal, float dist, float epsilon,
    std::vector<vec_t>& output )
{
  output.clear();

  if ( input.empty() )
    return;

  vec_t prev = input.back();
  float prev_signed = glm::dot( prev, normal ) - dist;
  bool prev_in = prev_signed >= -epsilon;

  for ( const vec_t& current : input )
  {
    const float current_signed = glm::dot( current, normal ) - dist;
    const bool current_in = current_signed >= -epsilon;
    if ( prev_in != current_in )
    {
      const float t = prev_signed / ( prev_signed - current_signed );
      output.push_back( prev + ( current - prev ) * t );
    }

    if ( current_in )
      output.push_back( current );

    prev = current;
    prev_signed = current_signed;
    prev_in = current_in;
  }
}
