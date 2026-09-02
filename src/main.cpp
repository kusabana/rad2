#include "bsp.h"
#include "entities.h"
#include "geometry.h"
#include "lights.h"
#include "log.h"
#include "luxels.h"
#include "patches.h"
#include "solve.h"
#include "visibility.h"

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <optional>
#include <vector>

namespace
{

struct cli_error : fatal_error
{
  using fatal_error::fatal_error;
};

enum class lighting_targets : uint8_t
{
  ldr_only,
  hdr_only,
  both,
};

constexpr uint32_t fast_sun_rays = 8;
constexpr uint32_t fast_sky_rays = 16;

struct cli_config
{
  std::filesystem::path bsp_path;
  std::filesystem::path output_path;

  int bounce_count = int( vrad_bounce_cap );
  bool fast = false;
  std::optional<uint32_t> sky_rays;

  lighting_targets targets = lighting_targets::both;

  bool list_devices = false;
  uint32_t device_id = 0;
};

template <typename number_t> number_t parse_number( std::string_view flag, std::string_view text )
{
  number_t value = 0;
  const auto result = std::from_chars( text.data(), text.data() + text.size(), value );
  if ( result.ec != std::errc() || result.ptr != text.data() + text.size() )
    throw cli_error( std::format( "{} needs a number, not '{}'", flag, text ) );

  return value;
}

cli_config parse_command_line( int argc, char** argv )
{
  cli_config config;
  bool have_bsp = false;

  for ( int index = 1; index < argc; ++index )
  {
    const std::string_view flag = argv[index];
    if ( !flag.empty() && flag[0] != '-' )
    {
      if ( have_bsp )
        throw cli_error( std::format( "unexpected argument '{}'", flag ) );

      config.bsp_path = flag;
      have_bsp = true;
      continue;
    }

    const auto value = [&]() -> std::string_view
    {
      if ( index + 1 >= argc )
        throw cli_error( std::format( "{} expects a value", flag ) );

      return argv[++index];
    };

    if ( flag == "--output" )
    {
      config.output_path = value();
    }
    else if ( flag == "--bounces" )
    {
      config.bounce_count = parse_number<int>( flag, value() );

      if ( config.bounce_count < 0 )
        throw cli_error( "--bounces must be >= 0" );
    }
    else if ( flag == "--sky-rays" )
    {
      config.sky_rays = parse_number<uint32_t>( flag, value() );

      if ( *config.sky_rays < 1 )
        throw cli_error( "--sky-rays must be >= 1" );
    }
    else if ( flag == "--fast" )
    {
      config.fast = true;
    }
    else if ( flag == "--ldr" )
    {
      config.targets = lighting_targets::ldr_only;
    }
    else if ( flag == "--hdr" )
    {
      config.targets = lighting_targets::hdr_only;
    }
    else if ( flag == "--device" )
    {
      const std::string_view device = value();
      if ( device == "list" )
        config.list_devices = true;
      else
        config.device_id = parse_number<uint32_t>( flag, device );
    }
    else
    {
      throw cli_error( std::format( "unknown flag '{}'", flag ) );
    }
  }

  if ( !have_bsp && !config.list_devices )
    throw cli_error( "no bsp file given" );

  if ( config.output_path.empty() )
    config.output_path = config.bsp_path;

  return config;
}

constexpr std::string_view usage = R"(usage: rad2 <map.bsp> [options]

output:
  --output <path>    write the compiled map here (default: in place)
  --ldr | --hdr      solve a single lighting target (default: both)

quality:
  --bounces <n>      maximum bounce count (default 100)
  --sky-rays <n>     sky rays per luxel sample (default 2592, matches vrad -final)
  --fast             minimum ray counts for quick previews

device:
  --device list      print devices and exit
  --device <id>      solve on the given SYCL device (default 0)
)";

void print_devices()
{
  const std::vector<compute_device_info> devices = list_devices();
  if ( devices.empty() )
    log_info( "no SYCL GPU devices found" );

  for ( const compute_device_info& device : devices )
  {
    log_info( "device {} {} [{}]{}", device.index, device.name, device.vendor,
        device.embree_supported ? "" : " (no Embree support)" );
  }
}

void run( const cli_config& config )
{
  if ( config.list_devices )
  {
    print_devices();
    return;
  }

  const auto run_start = std::chrono::steady_clock::now();

  bsp_file bsp( config.bsp_path );

  log_info( "loaded {} (map revision {})", config.bsp_path.string(), bsp.map_revision() );

  lighting_targets targets = config.targets;
  if ( bsp.version() == 19 && targets != lighting_targets::ldr_only )
  {
    if ( targets == lighting_targets::hdr_only )
      throw fatal_error( "bsp version 19 does not support HDR lighting" );

    log_warn( "bsp version 19 does not support HDR lighting" );
    targets = lighting_targets::ldr_only;
  }

  const entity_lump entities( bsp.entity_text );

  log_info( "{} entities", entities.entities().size() );

  std::vector<light_table> tables;
  if ( targets != lighting_targets::hdr_only )
    tables.emplace_back( bsp, entities, lighting_target::ldr );

  if ( targets != lighting_targets::ldr_only )
    tables.emplace_back( bsp, entities, lighting_target::hdr );

  if ( tables.size() == 2 && tables[0] == tables[1] )
    tables.pop_back();

  for ( const light_table& table : tables )
    log_info( "{} {} lights", table.lights.size(), target_name( table.target ) );

  progress geometry_progress( "scene geometry" );
  const scene_geometry geometry( bsp, entities );
  geometry_progress.finish();

  progress luxel_progress( "luxel grid" );
  const luxel_grid luxels( bsp, geometry );
  luxel_progress.finish();

  solve_settings settings;
  settings.bounce_cap = uint32_t( config.bounce_count );

  if ( config.fast )
  {
    settings.sun_samples = fast_sun_rays;
    settings.sky_samples = fast_sky_rays;
  }

  if ( config.sky_rays )
    settings.sky_samples = *config.sky_rays;

  patch_tree patches;
  transfer_tables transfers;

  if ( settings.bounce_cap > 0 )
  {
    progress tree_progress( "patch tree" );
    patches = patch_tree( bsp, geometry );
    tree_progress.finish();
    transfers = build_transfer_tables( bsp, patches );
  }

  std::vector<direct_layers> direct;
  for ( const light_table& table : tables )
    direct.push_back( direct_light( geometry, luxels, table, settings, config.device_id ) );

  if ( !transfers.receivers.empty() )
  {
    const std::vector<std::vector<vec3>> bounce = bounce_light(
        geometry, luxels, patches, transfers, direct, settings.bounce_cap, config.device_id );

    for ( size_t t = 0; t < tables.size(); ++t )
    {
      for ( size_t i = 0; i < direct[t].lightmap.size(); ++i )
        direct[t].lightmap[i] += bounce[t][i];
    }
  }

  progress output_progress( "writing lumps" );
  const auto write_target = [&]( size_t t, lighting_target target )
  {
    bsp.write_lighting( geometry, direct[t].lightmap, target );
    bsp.write_worldlights( tables[t], target );
  };

  if ( targets != lighting_targets::hdr_only )
    write_target( 0, lighting_target::ldr );

  if ( targets != lighting_targets::ldr_only )
    write_target( tables.size() - 1, lighting_target::hdr );

  bsp.write_vertnormals( geometry );

  bsp.save( config.output_path );
  output_progress.finish();

  log_info( "wrote {}", config.output_path.string() );
  log_info( "finished in {}",
      format_duration(
          std::chrono::duration<double>( std::chrono::steady_clock::now() - run_start ).count() ) );
}

} // namespace

int main( int argc, char** argv )
{
  cli_config config;

  try
  {
    config = parse_command_line( argc, argv );
  }
  catch ( const cli_error& error )
  {
    log_error( "{}", error.what() );
    std::cout << "\n" << usage;
    return 1;
  }

  int exit_code = 0;

  try
  {
    run( config );
  }
  catch ( const fatal_error& error )
  {
    log_error( "{}", error.what() );
    exit_code = 1;
  }
  catch ( const std::exception& error )
  {
    log_error( "unhandled exception: {}", error.what() );
    exit_code = 1;
  }

  const uint64_t warnings = log_warning_count();
  if ( warnings > 0 )
    log_info( "finished with {} warnings", warnings );

  return exit_code;
}
