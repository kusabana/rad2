#pragma once

#include "log.h"
#include "math.h"

#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

std::optional<float> parse_float( std::string_view text );

class entity
{
public:
  std::optional<std::string_view> value( std::string_view key ) const;

  std::string_view classname() const;
  float float_value( std::string_view key, float fallback = 0.0f ) const;
  int int_value( std::string_view key, int fallback = 0 ) const;
  vec3 vec3_value( std::string_view key, const vec3& fallback = vec3( 0.0f ) ) const;

  std::vector<std::pair<std::string, std::string>> pairs;
};

class entity_lump
{
public:
  explicit entity_lump( std::string_view text );

  std::span<const entity> entities() const
  {
    return entities_;
  }

  const entity* find_by_targetname( std::string_view name ) const;

private:
  std::vector<entity> entities_;
};
