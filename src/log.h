#pragma once

#include <chrono>
#include <cstdint>
#include <format>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

struct fatal_error : std::runtime_error
{
  using std::runtime_error::runtime_error;
};

enum class log_level : uint8_t
{
  info,
  warn,
  error,
};

void log_write( log_level level, std::string_view message );

uint64_t log_warning_count();

std::string format_duration( double seconds );

class progress
{
public:
  explicit progress( std::string label );
  ~progress();

  progress( const progress& ) = delete;
  progress& operator=( const progress& ) = delete;

  void set( size_t done, size_t total );

  void finish();

private:
  static constexpr int dot_count = 4;

  void write( std::string_view text );

  std::string label_;
  std::optional<std::chrono::steady_clock::time_point> start_;
  int dots_ = 0;
};

template <class... args_t> void log_info( std::format_string<args_t...> fmt, args_t&&... args )
{
  log_write( log_level::info, std::format( fmt, std::forward<args_t>( args )... ) );
}

template <class... args_t> void log_warn( std::format_string<args_t...> fmt, args_t&&... args )
{
  log_write( log_level::warn, std::format( fmt, std::forward<args_t>( args )... ) );
}

template <class... args_t> void log_error( std::format_string<args_t...> fmt, args_t&&... args )
{
  log_write( log_level::error, std::format( fmt, std::forward<args_t>( args )... ) );
}
