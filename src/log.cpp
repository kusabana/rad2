#include "log.h"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <mutex>

namespace
{

std::atomic<uint64_t> g_warning_count = 0;

std::mutex g_write_mutex;
const progress* g_open_line = nullptr;

constexpr std::string_view level_prefix[] = { "", "[warn] ", "[error] " };

double seconds_since( std::chrono::steady_clock::time_point start )
{
  return std::chrono::duration<double>( std::chrono::steady_clock::now() - start ).count();
}

void close_line()
{
  if ( g_open_line )
  {
    std::fputc( '\n', stdout );
    std::fflush( stdout );
    g_open_line = nullptr;
  }
}

} // namespace

uint64_t log_warning_count()
{
  return g_warning_count.load( std::memory_order_relaxed );
}

std::string format_duration( double seconds )
{
  if ( seconds < 60.0 )
    return std::format( "{:.1f}s", seconds );

  const int whole = int( seconds );
  if ( whole < 3600 )
    return std::format( "{}m {:02d}s", whole / 60, whole % 60 );

  return std::format( "{}h {:02d}m", whole / 3600, ( whole % 3600 ) / 60 );
}

void log_write( log_level level, std::string_view message )
{
  if ( level == log_level::warn )
    g_warning_count.fetch_add( 1, std::memory_order_relaxed );

  const std::string_view prefix = level_prefix[size_t( level )];

  std::lock_guard<std::mutex> lock( g_write_mutex );
  close_line();

  std::FILE* stream = ( level == log_level::error ) ? stderr : stdout;
  std::fwrite( prefix.data(), 1, prefix.size(), stream );
  std::fwrite( message.data(), 1, message.size(), stream );
  std::fputc( '\n', stream );
  std::fflush( stream );
}

progress::progress( std::string label )
    : label_( std::move( label ) ), start_( std::chrono::steady_clock::now() )
{
  std::lock_guard<std::mutex> lock( g_write_mutex );
  write( "" );
}

progress::~progress()
{
  finish();
}

void progress::write( std::string_view text )
{
  if ( g_open_line != this )
  {
    close_line();
    const std::string head = label_ + std::string( size_t( dots_ ), '.' );
    std::fwrite( head.data(), 1, head.size(), stdout );
    g_open_line = this;
  }

  std::fwrite( text.data(), 1, text.size(), stdout );
  std::fflush( stdout );

  if ( text.ends_with( '\n' ) )
    g_open_line = nullptr;
}

void progress::set( size_t done, size_t total )
{
  std::lock_guard<std::mutex> lock( g_write_mutex );

  const int reached =
      total > 0 ? int( std::min( size_t( dot_count ) * done / total, size_t( dot_count ) ) )
                : dot_count;

  for ( ; dots_ < reached; ++dots_ )
    write( "." );
}

void progress::finish()
{
  if ( !start_ )
    return;

  std::lock_guard<std::mutex> lock( g_write_mutex );

  for ( ; dots_ < dot_count; ++dots_ )
    write( "." );

  write( std::format( " {}\n", format_duration( seconds_since( *start_ ) ) ) );
  start_.reset();
}
