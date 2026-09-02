#pragma once

#include "bsp.h"
#include "entities.h"
#include "math.h"

#include <string_view>
#include <vector>

enum class lighting_target : uint8_t
{
  ldr,
  hdr,
};

constexpr std::string_view target_name( lighting_target target )
{
  return target == lighting_target::ldr ? "ldr" : "hdr";
}

// utils/vrad/lightmap.cpp:1176
constexpr float no_cap_distance = 1.0e22f;

struct light : dworldlight
{
  light() : dworldlight{}
  {
    type = emit_type::point;
    normal = vec3( 0.0f, 0.0f, 1.0f );
  }

  float start_fade = 0.0f;
  float end_fade = -1.0f;
  float cap_dist = no_cap_distance;

  bool operator==( const light& ) const = default;
};

struct sky_light
{
  vec3 sun_normal = vec3( 0.0f );
  float sun_extent_sin = 0.0f;
  vec3 sun_intensity = vec3( 0.0f );
  vec3 ambient_intensity = vec3( 0.0f );
  vec3 cam_origin = vec3( 0.0f );
  float world_to_sky = 0.0f;
  int32_t cam_area = -1;
};

class light_table
{
public:
  light_table( const bsp_file& bsp, const entity_lump& entities, lighting_target target );

  bool operator==( const light_table& other ) const
  {
    return lights == other.lights && sun_angular_extent_sin == other.sun_angular_extent_sin &&
           sky_camera_origin == other.sky_camera_origin &&
           sky_camera_scale == other.sky_camera_scale;
  }

  lighting_target target;
  std::vector<light> lights;
  float sun_angular_extent_sin = 0.0f;
  vec3 sky_camera_origin = vec3( 0.0f );
  float sky_camera_scale = 0.0f;
  int32_t sky_camera_area = -1;
};
