#include <sycl/sycl.hpp>

#include "solve.h"

#include "filter.h"
#include "sample_kernels.h"
#include "transfer_kernels.h"

#include <embree4/rtcore.h>

#include <cstring>
#include <format>
#include <numeric>

namespace
{

// luxels per launch
constexpr uint32_t slice_luxels = 1u << 16;

// receivers per transfer batch
constexpr uint32_t batch_size = 4096;

constexpr double reserve_headroom = 1.15;
constexpr double grow_factor = 1.5;

// utils/vrad/vrad.cpp:104
constexpr int resample_passes = 4;

sycl::device chosen_device( uint32_t index )
{
  const std::vector<sycl::device> devices =
      sycl::device::get_devices( sycl::info::device_type::gpu );
  if ( index >= devices.size() || !rtcIsSYCLDeviceSupported( devices[index] ) )
    throw fatal_error( std::format( "device {} is not usable", index ) );

  return devices[index];
}

void attach_triangles( RTCDevice device, RTCScene scene, uint32_t geom_id, uint32_t mask,
    const std::vector<vec3>& vertex_pool, const std::vector<uint32_t>& indices )
{
  if ( indices.empty() )
    return;

  RTCGeometry geometry = rtcNewGeometry( device, RTC_GEOMETRY_TYPE_TRIANGLE );

  vec3* vertices = static_cast<vec3*>( rtcSetNewGeometryBuffer( geometry, RTC_BUFFER_TYPE_VERTEX, 0,
      RTC_FORMAT_FLOAT3, sizeof( vec3 ), vertex_pool.size() ) );
  std::memcpy( vertices, vertex_pool.data(), vertex_pool.size() * sizeof( vec3 ) );

  uint32_t* triangles = static_cast<uint32_t*>( rtcSetNewGeometryBuffer( geometry,
      RTC_BUFFER_TYPE_INDEX, 0, RTC_FORMAT_UINT3, sizeof( uint32_t ) * 3, indices.size() / 3 ) );
  std::memcpy( triangles, indices.data(), indices.size() * sizeof( uint32_t ) );

  rtcSetGeometryMask( geometry, mask );
  rtcCommitGeometry( geometry );
  rtcAttachGeometryByID( scene, geometry, geom_id );
  rtcReleaseGeometry( geometry );
}

template <typename element_t> class device_buffer
{
public:
  explicit device_buffer( sycl::queue& queue ) : queue_( &queue ) {}

  ~device_buffer()
  {
    release();
  }

  device_buffer( const device_buffer& ) = delete;
  device_buffer& operator=( const device_buffer& ) = delete;

  void reserve( size_t count )
  {
    grow( count, 0 );
  }

  void upload( const element_t* host, size_t count )
  {
    reserve( count );

    if ( count > 0 )
      queue_->memcpy( data_, host, count * sizeof( element_t ) ).wait();
  }

  void upload( std::span<const element_t> host )
  {
    upload( host.data(), host.size() );
  }

  void grow( size_t count, size_t used )
  {
    if ( count <= capacity_ )
      return;

    if ( used == 0 )
      release();

    element_t* moved = sycl::malloc_device<element_t>( count, *queue_ );
    if ( !moved )
    {
      throw fatal_error( std::format( "device allocation of {} MB failed",
          double( count * sizeof( element_t ) ) / ( 1024.0 * 1024.0 ) ) );
    }

    if ( used > 0 )
      queue_->memcpy( moved, data_, used * sizeof( element_t ) ).wait();

    release();
    data_ = moved;
    capacity_ = count;
  }

  element_t* data() const
  {
    return data_;
  }

  size_t capacity() const
  {
    return capacity_;
  }

private:
  void release()
  {
    if ( data_ )
    {
      queue_->wait();
      sycl::free( data_, *queue_ );
    }

    data_ = nullptr;
    capacity_ = 0;
  }

  sycl::queue* queue_;
  element_t* data_ = nullptr;
  size_t capacity_ = 0;
};

class device_scene
{
public:
  device_scene( const scene_geometry& geometry, uint32_t device_index )
      : queue( chosen_device( device_index ), sycl::property::queue::in_order() )
  {
    rtc_device_ = rtcNewSYCLDevice( queue.get_context(), nullptr );
    if ( !rtc_device_ )
      throw fatal_error( "rtcNewSYCLDevice failed" );

    scene_ = rtcNewScene( rtc_device_ );
    rtcSetSceneFlags( scene_, RTC_SCENE_FLAG_ROBUST );
    rtcSetSceneBuildQuality( scene_, RTC_BUILD_QUALITY_HIGH );

    attach_triangles( rtc_device_, scene_, geom::occluder_id, geom::occluder_mask,
        geometry.vertices, geometry.occluder_indices );
    attach_triangles( rtc_device_, scene_, geom::sky_id, geom::sky_mask, geometry.vertices,
        geometry.sky_indices );
    attach_triangles( rtc_device_, scene_, geom::face_lm_id, geom::face_lm_mask, geometry.vertices,
        geometry.face_lm_indices );

    rtcCommitScene( scene_ );
    traversable = rtcGetSceneTraversable( scene_ );
  }

  ~device_scene()
  {
    if ( scene_ )
      rtcReleaseScene( scene_ );

    if ( rtc_device_ )
      rtcReleaseDevice( rtc_device_ );
  }

  device_scene( const device_scene& ) = delete;
  device_scene& operator=( const device_scene& ) = delete;

  sycl::queue queue;
  RTCTraversable traversable = nullptr;

private:
  RTCDevice rtc_device_ = nullptr;
  RTCScene scene_ = nullptr;
};

sky_light sky_from( const light_table& table )
{
  sky_light sky;

  for ( const light& source : table.lights )
  {
    if ( source.type == emit_type::skylight )
    {
      sky.sun_normal = source.normal;
      sky.sun_intensity = source.intensity;
      sky.sun_extent_sin = table.sun_angular_extent_sin;
    }
    else if ( source.type == emit_type::skyambient )
    {
      sky.ambient_intensity = source.intensity;
    }
  }

  if ( table.sky_camera_scale > 0.0f )
  {
    sky.cam_origin = table.sky_camera_origin;
    sky.world_to_sky = 1.0f / table.sky_camera_scale;
    sky.cam_area = table.sky_camera_area;
  }

  return sky;
}

std::vector<light> point_lights( const light_table& table )
{
  std::vector<light> lights;

  for ( const light& source : table.lights )
  {
    if ( source.type == emit_type::point || source.type == emit_type::spotlight )
      lights.push_back( source );
  }

  return lights;
}

template <typename evaluate_t>
void sample_layer( device_scene& device, const sample_scene& scene, const kernel_params& params,
    evaluate_t evaluate, std::vector<vec3>& layer, progress& pass_progress )
{
  device_buffer<vec3> out_buffer( device.queue );
  out_buffer.reserve( scene.luxel_count );

  const ray_tracer tracer{ device.traversable };
  vec3* dev_out = out_buffer.data();

  for ( uint32_t first = 0; first < scene.luxel_count; first += slice_luxels )
  {
    const uint32_t count = glm::min( slice_luxels, scene.luxel_count - first );

    device.queue
        .parallel_for( sycl::range<1>( count ),
            [=]( sycl::id<1> index )
            {
              sample_kernel(
                  scene, tracer, first + uint32_t( index[0] ), params, dev_out, evaluate );
            } )
        .wait();
    pass_progress.set( first + count, scene.luxel_count );
  }

  device.queue.memcpy( layer.data(), out_buffer.data(), scene.luxel_count * sizeof( vec3 ) ).wait();
}

// utils/vrad/lightmap.cpp:2943
template <typename evaluate_t>
void resample_layer( device_scene& device, const sample_scene& scene, const kernel_params& params,
    evaluate_t evaluate, std::span<const uint32_t> flagged, std::vector<vec3>& layer )
{
  const uint32_t count = uint32_t( flagged.size() );
  const uint32_t positions = params.position_samples;
  const uint32_t slice = slice_luxels / positions;

  device_buffer<uint32_t> flagged_buffer( device.queue );
  device_buffer<vec3> slot_buffer( device.queue );
  device_buffer<uint8_t> used_buffer( device.queue );
  device_buffer<vec3> result_buffer( device.queue );
  flagged_buffer.upload( flagged );
  slot_buffer.reserve( slice_luxels );
  used_buffer.reserve( slice_luxels );
  result_buffer.reserve( count );

  const ray_tracer tracer{ device.traversable };
  vec3* dev_slots = slot_buffer.data();
  uint8_t* dev_used = used_buffer.data();

  for ( uint32_t first = 0; first < count; first += slice )
  {
    const uint32_t slice_count = glm::min( slice, count - first );
    const uint32_t* slice_flagged = flagged_buffer.data() + first;
    vec3* slice_results = result_buffer.data() + first;

    device.queue.parallel_for( sycl::range<1>( size_t( slice_count ) * positions ),
        [=]( sycl::id<1> index )
        {
          const uint32_t i = uint32_t( index[0] );
          resample_position_kernel( scene, tracer, slice_flagged, i / positions, i % positions,
              params, dev_slots, dev_used, evaluate );
        } );

    device.queue
        .parallel_for( sycl::range<1>( slice_count ),
            [=]( sycl::id<1> index )
            {
              resample_average_kernel( scene, tracer, slice_flagged, uint32_t( index[0] ), params,
                  dev_slots, dev_used, slice_results, evaluate );
            } )
        .wait();
  }

  std::vector<vec3> results( count );
  device.queue.memcpy( results.data(), result_buffer.data(), size_t( count ) * sizeof( vec3 ) )
      .wait();

  for ( uint32_t f = 0; f < count; ++f )
    layer[flagged[f]] = results[f];
}

class transfer_matrix
{
public:
  transfer_matrix( device_scene& device, const patch_tree& tree, const transfer_tables& tables );

  void gather( const std::vector<vec3>& emit_scaled, std::vector<vec3>& received );

private:
  void add_batch( const uint32_t* receivers, uint32_t count, uint32_t* row_lengths );
  void prefix_counts( device_buffer<uint32_t>& counts, uint32_t count );

  device_scene& device_;

  device_buffer<patch_node> nodes_{ device_.queue };
  device_buffer<vec3> winding_{ device_.queue };
  device_buffer<uint32_t> cluster_first_{ device_.queue };
  device_buffer<uint32_t> cluster_faces_{ device_.queue };
  device_buffer<vec4> root_plane_{ device_.queue };
  device_buffer<int32_t> face_root_{ device_.queue };
  transfer_scene scene_;

  device_buffer<uint32_t> batch_receivers_{ device_.queue };
  device_buffer<uint32_t> batch_offsets_{ device_.queue };
  device_buffer<uint32_t> candidate_node_{ device_.queue };
  device_buffer<float> candidate_trans_{ device_.queue };
  device_buffer<uint32_t> candidate_receiver_{ device_.queue };
  device_buffer<uint8_t> candidate_visible_{ device_.queue };
  device_buffer<uint32_t> batch_visible_{ device_.queue };
  device_buffer<float> batch_scale_{ device_.queue };
  std::vector<uint32_t> prefix_host_;

  device_buffer<uint32_t> matrix_node_{ device_.queue };
  device_buffer<float> matrix_factor_{ device_.queue };
  size_t transfer_count_ = 0;
  uint32_t receiver_count_ = 0;

  device_buffer<uint32_t> row_first_{ device_.queue };
  device_buffer<uint32_t> leaves_{ device_.queue };
  device_buffer<vec3> emit_{ device_.queue };
  device_buffer<vec3> received_{ device_.queue };
  uint32_t leaf_count_ = 0;
};

transfer_matrix::transfer_matrix(
    device_scene& device, const patch_tree& tree, const transfer_tables& tables )
    : device_( device ), receiver_count_( uint32_t( tables.receivers.size() ) )
{
  nodes_.upload( tree.nodes );
  winding_.upload( tree.winding );
  cluster_first_.upload( tables.cluster_first );
  cluster_faces_.upload( tables.cluster_faces );
  root_plane_.upload( tree.root_plane );
  face_root_.upload( tree.face_root );

  scene_.nodes = nodes_.data();
  scene_.winding = winding_.data();
  scene_.cluster_first = cluster_first_.data();
  scene_.cluster_faces = cluster_faces_.data();
  scene_.root_plane = root_plane_.data();
  scene_.face_root = face_root_.data();

  std::vector<uint32_t> row_first( tree.nodes.size() + 1 );
  std::vector<uint32_t> lengths( batch_size );
  progress transfer_progress( "transfers" );

  for ( size_t first = 0; first < tables.receivers.size(); first += batch_size )
  {
    const uint32_t count =
        uint32_t( glm::min( size_t( batch_size ), tables.receivers.size() - first ) );
    add_batch( tables.receivers.data() + first, count, lengths.data() );

    for ( uint32_t bi = 0; bi < count; ++bi )
      row_first[tables.receivers[first + bi] + 1] = lengths[bi];

    transfer_progress.set( first + count, tables.receivers.size() );
  }

  transfer_progress.finish();
  std::inclusive_scan( row_first.begin(), row_first.end(), row_first.begin() );

  row_first_.upload( row_first );
  leaves_.upload( tree.leaves );
  leaf_count_ = uint32_t( tree.leaves.size() );
  emit_.reserve( tree.nodes.size() );
  received_.reserve( tree.nodes.size() );
  device_.queue.memset( received_.data(), 0, tree.nodes.size() * sizeof( vec3 ) ).wait();
}

void transfer_matrix::add_batch( const uint32_t* receivers, uint32_t count, uint32_t* row_lengths )
{
  if ( count == 0 )
    return;

  batch_receivers_.reserve( count );
  batch_offsets_.reserve( size_t( count ) + 1 );
  device_.queue.memcpy( batch_receivers_.data(), receivers, count * sizeof( uint32_t ) );

  const transfer_scene scene = scene_;
  const uint32_t* dev_receivers = batch_receivers_.data();
  uint32_t* dev_offsets = batch_offsets_.data();

  device_.queue.parallel_for( sycl::range<1>( count ),
      [=]( sycl::id<1> index )
      {
        const uint32_t bi = uint32_t( index[0] );
        dev_offsets[bi + 1] =
            collect_transfer_candidates( scene, dev_receivers[bi], nullptr, nullptr );
      } );

  prefix_counts( batch_offsets_, count );
  const size_t candidates = prefix_host_[count];

  candidate_node_.reserve( candidates );
  candidate_trans_.reserve( candidates );
  candidate_receiver_.reserve( candidates );
  candidate_visible_.reserve( candidates );

  uint32_t* dev_node = candidate_node_.data();
  float* dev_trans = candidate_trans_.data();
  uint32_t* dev_candidate_receiver = candidate_receiver_.data();
  uint8_t* dev_visible = candidate_visible_.data();

  device_.queue.parallel_for( sycl::range<1>( count ),
      [=]( sycl::id<1> index )
      {
        const uint32_t bi = uint32_t( index[0] );
        collect_transfer_candidates(
            scene, dev_receivers[bi], dev_node + dev_offsets[bi], dev_trans + dev_offsets[bi] );

        for ( uint32_t k = dev_offsets[bi]; k < dev_offsets[bi + 1]; ++k )
          dev_candidate_receiver[k] = dev_receivers[bi];
      } );

  if ( candidates > 0 )
  {
    RTCTraversable traversable = device_.traversable;

    // one work item per candidate ray
    device_.queue.parallel_for( sycl::range<1>( candidates ),
        [=]( sycl::id<1> index )
        {
          const uint32_t k = uint32_t( index[0] );
          const ray_tracer tracer{ traversable };
          dev_visible[k] =
              transfer_visible( scene, tracer, dev_candidate_receiver[k], dev_node[k] );
        } );
  }

  batch_visible_.reserve( size_t( count ) + 1 );
  batch_scale_.reserve( count );
  uint32_t* dev_counts = batch_visible_.data();
  float* dev_scale = batch_scale_.data();

  device_.queue.parallel_for( sycl::range<1>( count ),
      [=]( sycl::id<1> index )
      {
        const uint32_t bi = uint32_t( index[0] );
        uint32_t kept = 0;
        float sum = 0.0f;

        for ( uint32_t k = dev_offsets[bi]; k < dev_offsets[bi + 1]; ++k )
        {
          if ( dev_visible[k] )
          {
            ++kept;
            sum += dev_trans[k];
          }
        }

        dev_counts[bi + 1] = kept;
        dev_scale[bi] = sum > glm::pi<float>() ? 1.0f / sum : 1.0f / glm::pi<float>();
      } );

  prefix_counts( batch_visible_, count );
  const size_t total = prefix_host_[count];

  for ( uint32_t bi = 0; bi < count; ++bi )
    row_lengths[bi] = prefix_host_[bi + 1] - prefix_host_[bi];

  if ( transfer_count_ == 0 )
  {
    const size_t estimate =
        size_t( double( total ) / double( count ) * double( receiver_count_ ) * reserve_headroom );
    matrix_node_.grow( estimate, 0 );
    matrix_factor_.grow( estimate, 0 );
  }

  if ( transfer_count_ + total > matrix_node_.capacity() )
  {
    const size_t wanted = glm::max(
        transfer_count_ + total, size_t( double( matrix_node_.capacity() ) * grow_factor ) );
    matrix_node_.grow( wanted, transfer_count_ );
    matrix_factor_.grow( wanted, transfer_count_ );
  }

  uint32_t* out_node = matrix_node_.data() + transfer_count_;
  float* out_factor = matrix_factor_.data() + transfer_count_;

  device_.queue.parallel_for( sycl::range<1>( count ),
      [=]( sycl::id<1> index )
      {
        const uint32_t bi = uint32_t( index[0] );
        uint32_t write = dev_counts[bi];

        for ( uint32_t k = dev_offsets[bi]; k < dev_offsets[bi + 1]; ++k )
        {
          if ( !dev_visible[k] )
            continue;

          out_node[write] = dev_node[k];
          out_factor[write] = dev_trans[k] * dev_scale[bi];
          ++write;
        }
      } );

  device_.queue.wait();
  transfer_count_ += total;
}

void transfer_matrix::prefix_counts( device_buffer<uint32_t>& counts, uint32_t count )
{
  uint32_t* dev_counts = counts.data();

  device_.queue.single_task(
      [=]()
      {
        dev_counts[0] = 0;

        for ( uint32_t bi = 0; bi < count; ++bi )
          dev_counts[bi + 1] += dev_counts[bi];
      } );

  prefix_host_.resize( size_t( count ) + 1 );
  device_.queue
      .memcpy( prefix_host_.data(), dev_counts, ( size_t( count ) + 1 ) * sizeof( uint32_t ) )
      .wait();
}

void transfer_matrix::gather( const std::vector<vec3>& emit_scaled, std::vector<vec3>& received )
{
  device_.queue.memcpy( emit_.data(), emit_scaled.data(), emit_scaled.size() * sizeof( vec3 ) );

  const uint32_t* row_first = row_first_.data();
  const uint32_t* node = matrix_node_.data();
  const float* factor = matrix_factor_.data();
  const uint32_t* leaves = leaves_.data();
  const vec3* emit = emit_.data();
  vec3* out = received_.data();

  device_.queue.parallel_for( sycl::range<1>( leaf_count_ ),
      [=]( sycl::id<1> index )
      {
        const uint32_t leaf = leaves[index[0]];
        vec3 sum( 0.0f );

        for ( uint32_t k = row_first[leaf]; k < row_first[leaf + 1]; ++k )
          sum += emit[node[k]] * factor[k];

        out[leaf] = sum;
      } );

  device_.queue.memcpy( received.data(), received_.data(), received.size() * sizeof( vec3 ) )
      .wait();
}

} // namespace

std::vector<compute_device_info> list_devices()
{
  std::vector<compute_device_info> result;
  uint32_t index = 0;

  for ( const sycl::device& device : sycl::device::get_devices( sycl::info::device_type::gpu ) )
  {
    result.push_back( { .index = index++,
        .name = device.get_info<sycl::info::device::name>(),
        .vendor = device.get_info<sycl::info::device::vendor>(),
        .embree_supported = rtcIsSYCLDeviceSupported( device ) } );
  }

  return result;
}

direct_layers direct_light( const scene_geometry& geometry, const luxel_grid& luxels,
    const light_table& table, const solve_settings& settings, uint32_t device_index )
{
  const std::string_view name = target_name( table.target );

  const sky_light sky = sky_from( table );
  const std::vector<light> lights = point_lights( table );
  if ( lights.empty() && max_component( sky.sun_intensity ) <= 0.0f &&
       max_component( sky.ambient_intensity ) <= 0.0f )
    log_warn( "no {} light sources", name );

  device_scene device( geometry, device_index );
  device_buffer<luxel> luxel_buffer( device.queue );
  device_buffer<face_info> face_buffer( device.queue );
  device_buffer<edge_plane> edge_plane_buffer( device.queue );
  device_buffer<light> light_buffer( device.queue );
  luxel_buffer.upload( luxels.samples );
  face_buffer.upload( geometry.faces );
  edge_plane_buffer.upload( geometry.edge_planes );
  light_buffer.upload( lights );

  const sample_scene scene{ .luxels = luxel_buffer.data(),
      .faces = face_buffer.data(),
      .edge_planes = edge_plane_buffer.data(),
      .lights = light_buffer.data(),
      .light_count = uint32_t( lights.size() ),
      .sky = sky,
      .luxel_count = uint32_t( luxels.samples.size() ) };

  kernel_params params;
  params.sun_samples = settings.sun_samples;
  params.sky_samples = settings.sky_samples;

  const size_t luxel_count = luxels.samples.size();
  std::vector<vec3> direct( luxel_count );
  std::vector<vec3> sky_light( luxel_count );
  const bool ambient = max_component( sky.ambient_intensity ) > 0.0f;

  {
    progress lights_progress( std::format( "[{}] lights", name ) );
    sample_layer( device, scene, params, direct_sample{}, direct, lights_progress );
  }

  if ( ambient )
  {
    progress sky_progress( std::format( "[{}] sky", name ) );
    sample_layer( device, scene, params, sky_sample{}, sky_light, sky_progress );
  }

  // extra sampling of steep gradients
  std::vector<uint8_t> processed( luxel_count );
  std::vector<vec3> combined( luxel_count );
  progress resample_progress( std::format( "[{}] resampling", name ) );

  for ( int pass = 0; pass < resample_passes; ++pass )
  {
    for ( size_t i = 0; i < luxel_count; ++i )
      combined[i] = direct[i] + sky_light[i];

    const std::vector<uint32_t> flagged =
        steep_gradient_luxels( geometry, luxels, combined, processed );
    if ( flagged.empty() )
      break;

    params.position_samples = settings.position_samples;
    resample_layer( device, scene, params, direct_sample{}, flagged, direct );

    if ( ambient )
    {
      params.position_samples = settings.sky_position_samples;
      resample_layer( device, scene, params, sky_sample{}, flagged, sky_light );
    }

    resample_progress.set( size_t( pass ) + 1, resample_passes );
  }

  resample_progress.finish();

  progress filter_progress( std::format( "[{}] filter", name ) );
  std::vector<vec3> raw = direct;
  radial_filter( geometry, luxels, direct );

  if ( ambient )
  {
    for ( size_t i = 0; i < luxel_count; ++i )
      raw[i] += sky_light[i];

    radial_filter( geometry, luxels, sky_light );

    for ( size_t i = 0; i < luxel_count; ++i )
      direct[i] += sky_light[i];
  }

  filter_progress.finish();
  return { name, std::move( direct ), std::move( raw ) };
}

std::vector<std::vector<vec3>> bounce_light( const scene_geometry& geometry,
    const luxel_grid& luxels, const patch_tree& tree, const transfer_tables& tables,
    std::span<const direct_layers> direct, uint32_t bounce_cap, uint32_t device_index )
{
  device_scene device( geometry, device_index );
  transfer_matrix matrix( device, tree, tables );

  const size_t count = tree.nodes.size();
  std::vector<vec3> scaled( count );
  std::vector<vec3> received( count );
  std::vector<std::vector<vec3>> result;

  for ( const direct_layers& layer : direct )
  {
    progress bounce_progress( std::format( "[{}] bounces", layer.name ) );
    std::vector<vec3> emit = tree.add_direct_light( geometry, luxels, layer.raw );
    std::vector<vec3> total( count );

    for ( uint32_t pass = 1; pass <= bounce_cap; ++pass )
    {
      for ( size_t i = 0; i < count; ++i )
        scaled[i] = emit[i] * geometry.faces[tree.nodes[i].face].reflectivity;

      matrix.gather( scaled, received );

      const vec3 added = tree.collect_light( received, emit, total );
      if ( added.x < 1.0f && added.y < 1.0f && added.z < 1.0f )
        break;
    }

    std::vector<vec3> delta( luxels.samples.size() );
    tree.splat_bounce( geometry, total, delta, bounce_progress );
    bounce_progress.finish();
    result.push_back( std::move( delta ) );
  }

  return result;
}
