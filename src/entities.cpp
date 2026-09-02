#include "entities.h"

#include <algorithm>
#include <cctype>
#include <charconv>

namespace
{

bool equals_ci( std::string_view a, std::string_view b )
{
  const auto lower = []( char c )
  {
    return std::tolower( uint8_t( c ) );
  };
  return std::ranges::equal( a, b, {}, lower, lower );
}

std::optional<float> parse_leading_float( std::string_view& text )
{
  text.remove_prefix( std::min( text.find_first_not_of( " \t" ), text.size() ) );

  float value = 0.0f;
  const std::from_chars_result result =
      std::from_chars( text.data(), text.data() + text.size(), value );
  if ( result.ec != std::errc() )
    return std::nullopt;

  text.remove_prefix( size_t( result.ptr - text.data() ) );
  return value;
}

} // namespace

std::optional<float> parse_float( std::string_view text )
{
  return parse_leading_float( text );
}

std::optional<std::string_view> entity::value( std::string_view key ) const
{
  for ( const auto& [pair_key, pair_value] : pairs )
  {
    if ( equals_ci( pair_key, key ) )
      return std::string_view( pair_value );
  }

  return std::nullopt;
}

std::string_view entity::classname() const
{
  return value( "classname" ).value_or( "" );
}

float entity::float_value( std::string_view key, float fallback ) const
{
  const std::optional<std::string_view> text = value( key );
  if ( !text )
    return fallback;

  return parse_float( *text ).value_or( fallback );
}

int entity::int_value( std::string_view key, int fallback ) const
{
  return int( float_value( key, float( fallback ) ) );
}

vec3 entity::vec3_value( std::string_view key, const vec3& fallback ) const
{
  std::optional<std::string_view> text = value( key );
  if ( !text )
    return fallback;

  vec3 result;

  for ( int c = 0; c < 3; ++c )
  {
    const std::optional<float> component = parse_leading_float( *text );
    if ( !component )
      return fallback;

    result[c] = *component;
  }

  return result;
}

entity_lump::entity_lump( std::string_view text )
{
  constexpr std::string_view blank = " \t\r\n";
  constexpr size_t npos = std::string_view::npos;

  const size_t nul = text.find( '\0' );
  if ( nul != npos )
    text = text.substr( 0, nul );

  size_t cursor = text.find( '{' );

  while ( cursor != npos )
  {
    entity current;
    bool closed = false;
    cursor = text.find_first_not_of( blank, cursor + 1 );

    while ( cursor != npos )
    {
      if ( text[cursor] == '}' )
      {
        closed = true;
        ++cursor;
        break;
      }

      const size_t key_end = text[cursor] == '"' ? text.find( '"', cursor + 1 ) : npos;
      const size_t value_begin =
          key_end == npos ? npos : text.find_first_not_of( blank, key_end + 1 );
      const size_t value_end = value_begin != npos && text[value_begin] == '"'
                                   ? text.find( '"', value_begin + 1 )
                                   : npos;

      if ( value_end == npos )
        break;

      current.pairs.emplace_back( std::string( text.substr( cursor + 1, key_end - cursor - 1 ) ),
          std::string( text.substr( value_begin + 1, value_end - value_begin - 1 ) ) );
      cursor = text.find_first_not_of( blank, value_end + 1 );
    }

    if ( closed )
      entities_.push_back( std::move( current ) );
    else
      log_warn( "malformed block in the entity lump" );

    cursor = text.find( '{', cursor );
  }
}

const entity* entity_lump::find_by_targetname( std::string_view name ) const
{
  for ( const entity& candidate : entities_ )
  {
    const std::optional<std::string_view> targetname = candidate.value( "targetname" );
    if ( targetname && equals_ci( *targetname, name ) )
      return &candidate;
  }

  return nullptr;
}
