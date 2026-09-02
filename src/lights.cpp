#include "lights.h"

#include <algorithm>
#include <cstdio>
#include <string>

namespace
{

// public/bspfile.h:895
constexpr float angle_up = -1.0f;
constexpr float angle_down = -2.0f;

// utils/vrad/lightmap.cpp:1186
constexpr float zero_percent_ratio = 256.0f;

// utils/vrad/lightmap.cpp:1211
constexpr float fade_start_blend = 0.75f;

// utils/vrad/lightmap.cpp:1226
constexpr float end_fade_scale = 10.0f;

// utils/vrad/lightmap.cpp:1253
constexpr float attenuation_reference = 100.0f;

// utils/vrad/lightmap.cpp:1273
constexpr float default_inner_cone = 10.0f;

float flerp( float f1, float f2, float i1, float i2, float x )
{
  return f1 + ( f2 - f1 ) * ( x - i1 ) / ( i2 - i1 );
}

// mathlib/mathlib_base.cpp:1345
bool solve_inverse_quadratic(
    float x1, float y1, float x2, float y2, float x3, float y3, float& a, float& b, float& c )
{
  const float det = ( x1 - x2 ) * ( x1 - x3 ) * ( x2 - x3 );
  if ( det == 0.0f )
    return false;

  a = ( x3 * ( -y1 + y2 ) + x2 * ( y1 - y3 ) + x1 * ( -y2 + y3 ) ) / det;
  b = ( x3 * x3 * ( y1 - y2 ) + x1 * x1 * ( y2 - y3 ) + x2 * x2 * ( -y1 + y3 ) ) / det;
  c = ( x1 * x3 * ( -x1 + x3 ) * y2 + x2 * x2 * ( x3 * y1 - x1 * y3 ) +
          x2 * ( -( x3 * x3 * y1 ) + x1 * x1 * y3 ) ) /
      det;

  return true;
}

// mathlib/mathlib_base.cpp:1362
bool solve_inverse_quadratic_monotonic(
    float x1, float y1, float x2, float y2, float x3, float y3, float& a, float& b, float& c )
{
  if ( x1 > x2 )
  {
    std::swap( x1, x2 );
    std::swap( y1, y2 );
  }

  if ( x2 > x3 )
  {
    std::swap( x2, x3 );
    std::swap( y2, y3 );
  }

  if ( x1 > x2 )
  {
    std::swap( x1, x2 );
    std::swap( y1, y2 );
  }

  for ( float blend = 0.0f; blend <= 1.0f; blend += 0.05f )
  {
    const float mid = ( 1.0f - blend ) * y2 + blend * flerp( y1, y3, x1, x3, x2 );
    if ( !solve_inverse_quadratic( x1, y1, x2, mid, x3, y3, a, b, c ) )
      return false;

    const float derivative = 2.0f * a + b;

    if ( y1 < y2 && y2 < y3 )
    {
      if ( derivative >= 0.0f )
        return true;
    }
    else if ( y1 > y2 && y2 > y3 )
    {
      if ( derivative <= 0.0f )
        return true;
    }
    else
    {
      return true;
    }
  }

  return true;
}

// public/map_utils.cpp:13
vec3 light_normal_from_props( const vec3& angles, float angle, float pitch )
{
  vec3 output( 0.0f );

  if ( angle == angle_up )
  {
    output = vec3( 0.0f, 0.0f, 1.0f );
  }
  else if ( angle == angle_down )
  {
    output = vec3( 0.0f, 0.0f, -1.0f );
  }
  else
  {
    if ( angle == 0.0f )
      angle = angles.y;

    output.x = glm::cos( glm::radians( angle ) );
    output.y = glm::sin( glm::radians( angle ) );
    output.z = 0.0f;
  }

  if ( pitch == 0.0f )
    pitch = angles.x;

  output.z = glm::sin( glm::radians( pitch ) );
  output.x *= glm::cos( glm::radians( pitch ) );
  output.y *= glm::cos( glm::radians( pitch ) );

  return output;
}

std::optional<vec3> light_for_string( std::string_view text, lighting_target target )
{
  double r = 0, g = 0, b = 0, scaler = 0;
  double r_hdr = 0, g_hdr = 0, b_hdr = 0, scaler_hdr = 0;

  const std::string copy( text );
  int arg_count = std::sscanf( copy.c_str(), "%lf %lf %lf %lf %lf %lf %lf %lf", &r, &g, &b, &scaler,
      &r_hdr, &g_hdr, &b_hdr, &scaler_hdr );

  if ( arg_count == 8 )
  {
    if ( target == lighting_target::hdr )
    {
      r = r_hdr;
      g = g_hdr;
      b = b_hdr;
      scaler = scaler_hdr;
    }

    arg_count = 4;
  }

  if ( r < 0.0 || g < 0.0 || b < 0.0 || scaler < 0.0 )
    return std::nullopt;

  vec3 intensity( 0.0f );
  intensity.x = float( glm::pow( r / double( byte_max ), light_gamma ) * double( byte_max ) );

  switch ( arg_count )
  {
  case 1:
    intensity.y = intensity.z = intensity.x;
    break;

  case 3:
  case 4:
    intensity.y = float( glm::pow( g / double( byte_max ), light_gamma ) * double( byte_max ) );
    intensity.z = float( glm::pow( b / double( byte_max ), light_gamma ) * double( byte_max ) );

    if ( arg_count == 4 )
      intensity *= float( scaler / double( byte_max ) );

    break;

  default:
    return std::nullopt;
  }

  return intensity;
}

std::optional<vec3> light_for_key( const entity& e, std::string_view key, lighting_target target )
{
  const std::optional<std::string_view> text = e.value( key );
  if ( !text )
    return std::nullopt;

  return light_for_string( *text, target );
}

// utils/vrad/lightmap.cpp:1124
void parse_light_generic(
    const entity& e, const entity_lump& entities, lighting_target target, light& source )
{
  std::optional<vec3> intensity;

  if ( target == lighting_target::hdr )
    intensity = light_for_key( e, "_lightHDR", target );

  if ( !intensity )
    intensity = light_for_key( e, "_light", target );

  source.intensity = intensity.value_or( vec3( 0.0f ) );

  const std::optional<std::string_view> targetname = e.value( "target" );
  if ( targetname && !targetname->empty() )
  {
    const entity* target_entity = entities.find_by_targetname( *targetname );
    if ( !target_entity )
    {
      log_warn( "light at ({} {} {}) has missing target", int( source.origin.x ),
          int( source.origin.y ), int( source.origin.z ) );
    }
    else
    {
      const vec3 dest = target_entity->vec3_value( "origin" );
      source.normal = safe_normalize( dest - source.origin );
    }
  }
  else
  {
    const vec3 angles = e.vec3_value( "angles" );
    const float pitch = e.float_value( "pitch" );
    const float angle = e.float_value( "angle" );
    source.normal = light_normal_from_props( angles, angle, pitch );
  }

  if ( target == lighting_target::hdr )
    source.intensity *= e.float_value( "_lightscaleHDR", 1.0f );
}

// utils/vrad/lightmap.cpp:1171
void set_light_falloff_params( const entity& e, light& source )
{
  const float d50 = e.float_value( "_fifty_percent_distance" );

  source.start_fade = 0.0f;
  source.end_fade = -1.0f;
  source.cap_dist = no_cap_distance;

  if ( d50 != 0.0f )
  {
    float d0 = e.float_value( "_zero_percent_distance" );
    if ( d0 < d50 )
    {
      log_warn( "light has _fifty_percent_distance {} but _zero_percent_distance {}", d50, d0 );
      d0 = 2.0f * d50;
    }

    float a = 0.0f;
    float b = 1.0f;
    float c = 0.0f;

    if ( !solve_inverse_quadratic_monotonic(
             0.0f, 1.0f, d50, 2.0f, d0, zero_percent_ratio, a, b, c ) )
      log_warn( "can't solve falloff quadratic for light ({} {})", d50, d0 );

    const float v50 = c + d50 * ( b + d50 * a );
    const float scale = 2.0f / v50;
    a *= scale;
    b *= scale;
    c *= scale;

    source.quadratic_attn = a;
    source.linear_attn = b;
    source.constant_attn = c;

    if ( e.int_value( "_hardfalloff" ) != 0 )
    {
      source.end_fade = d0;
      source.start_fade = fade_start_blend * d0 + ( 1.0f - fade_start_blend ) * d50;
    }
    else
    {
      if ( glm::abs( a ) > 0.0f )
      {
        const float max_dist = b / ( -2.0f * a );
        if ( max_dist > 0.0f )
        {
          source.cap_dist = max_dist;
          source.start_fade = max_dist;
          source.end_fade = end_fade_scale * max_dist;
        }
      }
    }
  }
  else
  {
    source.constant_attn = e.float_value( "_constant_attn" );
    source.linear_attn = e.float_value( "_linear_attn" );
    source.quadratic_attn = e.float_value( "_quadratic_attn" );
    source.radius = e.float_value( "_distance" );

    if ( source.constant_attn < equal_epsilon )
      source.constant_attn = 0.0f;

    if ( source.linear_attn < equal_epsilon )
      source.linear_attn = 0.0f;

    if ( source.quadratic_attn < equal_epsilon )
      source.quadratic_attn = 0.0f;

    if ( source.constant_attn < equal_epsilon && source.linear_attn < equal_epsilon &&
         source.quadratic_attn < equal_epsilon )
    {
      source.constant_attn = 1.0f;
    }

    const float ratio = source.constant_attn + attenuation_reference * source.linear_attn +
                        attenuation_reference * attenuation_reference * source.quadratic_attn;

    if ( ratio > 0.0f )
      source.intensity *= ratio;
  }
}

light parse_light_point( const entity& e, const entity_lump& entities, lighting_target target )
{
  light source;
  source.origin = e.vec3_value( "origin" );
  parse_light_generic( e, entities, target, source );
  set_light_falloff_params( e, source );
  return source;
}

// utils/vrad/lightmap.cpp:1261
light parse_light_spot( const entity& e, const entity_lump& entities, lighting_target target )
{
  light source = parse_light_point( e, entities, target );
  source.type = emit_type::spotlight;
  source.stopdot = e.float_value( "_inner_cone" );

  if ( source.stopdot == 0.0f )
    source.stopdot = default_inner_cone;

  source.stopdot2 = e.float_value( "_cone" );

  if ( source.stopdot2 == 0.0f )
    source.stopdot2 = source.stopdot;

  if ( source.stopdot2 < source.stopdot )
    source.stopdot2 = source.stopdot;

  if ( source.stopdot == 180.0f && source.stopdot2 == 180.0f )
  {
    source.stopdot = source.stopdot2 = 0.0f;
    source.type = emit_type::point;
    source.exponent = 0.0f;
  }
  else
  {
    if ( source.stopdot > 90.0f )
      source.stopdot = 90.0f;

    if ( source.stopdot2 > 90.0f )
      source.stopdot2 = 90.0f;

    source.stopdot2 = glm::cos( glm::radians( source.stopdot2 ) );
    source.stopdot = glm::cos( glm::radians( source.stopdot ) );
    source.exponent = e.float_value( "_exponent" );
  }

  return source;
}

} // namespace

light_table::light_table( const bsp_file& bsp, const entity_lump& entities, lighting_target target )
    : target( target )
{
  for ( const entity& e : entities.entities() )
  {
    const std::string_view classname = e.classname();
    if ( classname == "sky_camera" )
    {
      const float scale = e.float_value( "scale" );
      if ( scale > 0.0f )
      {
        sky_camera_origin = e.vec3_value( "origin" );
        sky_camera_scale = scale;
      }

      continue;
    }

    if ( !classname.starts_with( "light" ) )
      continue;

    if ( classname == "light_dynamic" )
      continue;

    if ( classname == "light_spot" )
    {
      lights.push_back( parse_light_spot( e, entities, target ) );
    }
    else if ( classname == "light_environment" )
    {
      light sun;
      sun.origin = e.vec3_value( "origin" );
      parse_light_generic( e, entities, target, sun );

      const std::optional<std::string_view> spread = e.value( "SunSpreadAngle" );
      if ( spread )
        sun_angular_extent_sin =
            glm::sin( glm::radians( parse_float( *spread ).value_or( 0.0f ) ) );

      if ( std::ranges::any_of(
               lights, []( const light& source ) { return source.type == emit_type::skylight; } ) )
        continue; // TODO: maybe warn the user that we've ignored a light_environment

      sun.type = emit_type::skylight;
      lights.push_back( sun );

      light ambient;
      ambient.type = emit_type::skyambient;
      ambient.origin = sun.origin;

      std::optional<vec3> ambient_intensity;

      if ( target == lighting_target::hdr )
        ambient_intensity = light_for_key( e, "_ambientHDR", target );

      if ( !ambient_intensity )
        ambient_intensity = light_for_key( e, "_ambient", target );

      ambient.intensity = ambient_intensity.value_or( sun.intensity * 0.5f );

      if ( target == lighting_target::hdr )
        ambient.intensity *= e.float_value( "_AmbientScaleHDR", 1.0f );

      lights.push_back( ambient );
    }
    else if ( classname == "light" )
    {
      lights.push_back( parse_light_point( e, entities, target ) );
    }
    else
    {
      log_warn( "unsupported light entity '{}'", classname );
      continue;
    }

    if ( e.int_value( "style" ) != 0 )
    {
      // TODO: handle styled lights
    }
  }

  if ( sky_camera_scale > 0.0f )
    sky_camera_area = bsp.leaf_at( sky_camera_origin ).area();
}
