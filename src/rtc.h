#pragma once

#include <sycl/sycl.hpp>

#include "log.h"
#include "math.h"

#include <embree4/rtcore.h>

#include <cstdint>
#include <cstring>
#include <span>

namespace rtc
{
using ray_hit = RTCRayHit;
using feature_flags = RTCFeatureFlags;
using scene_flags = RTCSceneFlags;
using build_quality = RTCBuildQuality;

struct ray : RTCRay
{
  ray( const vec3& origin, const vec3& direction, float t_min, float t_max, uint32_t ray_mask )
      : RTCRay{ .org_x = origin.x,
            .org_y = origin.y,
            .org_z = origin.z,
            .tnear = t_min,
            .dir_x = direction.x,
            .dir_y = direction.y,
            .dir_z = direction.z,
            .time = 0.0f,
            .tfar = t_max,
            .mask = ray_mask,
            .id = 0,
            .flags = 0 }
  {
  }
};

inline bool is_supported( const sycl::device& device )
{
  return rtcIsSYCLDeviceSupported( device );
}

class device
{
public:
  explicit device( const sycl::context& context ) : handle_( rtcNewSYCLDevice( context, nullptr ) )
  {
    if ( !handle_ )
      throw fatal_error( "rtcNewSYCLDevice failed" );
  }

  ~device()
  {
    rtcReleaseDevice( handle_ );
  }

  device( const device& ) = delete;
  device& operator=( const device& ) = delete;

  RTCDevice handle() const
  {
    return handle_;
  }

private:
  RTCDevice handle_;
};

class traversable
{
public:
  traversable() = default;

  explicit traversable( RTCTraversable handle ) : handle_( handle ) {}

  // returns the geometry id of the closest hit, or RTC_INVALID_GEOMETRY_ID
  uint32_t intersect( const ray& query, feature_flags features ) const
  {
    RTCIntersectArguments args;
    rtcInitIntersectArguments( &args );
    args.feature_mask = features;

    ray_hit hit = {};
    hit.ray = query;
    hit.hit.geomID = RTC_INVALID_GEOMETRY_ID;
    hit.hit.primID = RTC_INVALID_GEOMETRY_ID;

    rtcTraversableIntersect1( handle_, &hit, &args );
    return hit.hit.geomID;
  }

  bool occluded( ray query, feature_flags features ) const
  {
    RTCOccludedArguments args;
    rtcInitOccludedArguments( &args );
    args.feature_mask = features;

    rtcTraversableOccluded1( handle_, &query, &args );
    return query.tfar < 0.0f;
  }

private:
  RTCTraversable handle_ = nullptr;
};

class scene
{
public:
  explicit scene( const device& owner )
      : device_( owner.handle() ), handle_( rtcNewScene( device_ ) )
  {
    if ( !handle_ )
      throw fatal_error( "rtcNewScene failed" );
  }

  ~scene()
  {
    rtcReleaseScene( handle_ );
  }

  scene( const scene& ) = delete;
  scene& operator=( const scene& ) = delete;

  void set_flags( scene_flags flags )
  {
    rtcSetSceneFlags( handle_, flags );
  }

  void set_build_quality( build_quality quality )
  {
    rtcSetSceneBuildQuality( handle_, quality );
  }

  void attach_triangles( uint32_t geom_id, uint32_t mask, std::span<const vec3> vertices,
      std::span<const uint32_t> indices )
  {
    if ( indices.empty() )
      return;

    RTCGeometry geometry = rtcNewGeometry( device_, RTC_GEOMETRY_TYPE_TRIANGLE );

    void* vertex_buffer = rtcSetNewGeometryBuffer(
        geometry, RTC_BUFFER_TYPE_VERTEX, 0, RTC_FORMAT_FLOAT3, sizeof( vec3 ), vertices.size() );
    std::memcpy( vertex_buffer, vertices.data(), vertices.size_bytes() );

    void* index_buffer = rtcSetNewGeometryBuffer( geometry, RTC_BUFFER_TYPE_INDEX, 0,
        RTC_FORMAT_UINT3, sizeof( uint32_t ) * 3, indices.size() / 3 );
    std::memcpy( index_buffer, indices.data(), indices.size_bytes() );

    rtcSetGeometryMask( geometry, mask );
    rtcCommitGeometry( geometry );
    rtcAttachGeometryByID( handle_, geometry, geom_id );
    rtcReleaseGeometry( geometry );
  }

  void commit()
  {
    rtcCommitScene( handle_ );
  }

  rtc::traversable traversable() const
  {
    return rtc::traversable( rtcGetSceneTraversable( handle_ ) );
  }

private:
  RTCDevice device_;
  RTCScene handle_;
};

} // namespace rtc
