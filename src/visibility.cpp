#include "visibility.h"

#include <bit>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace
{

// public/bspfile.h:903
constexpr size_t vis_count_size = sizeof( int32_t );
constexpr size_t vis_offsets_size = 2 * sizeof( int32_t );

// run length decoding of one pvs row
void decompress_row( const uint8_t* in, const uint8_t* end, uint8_t* out, size_t row_bytes )
{
  size_t o = 0;

  while ( o < row_bytes && in < end )
  {
    if ( *in )
    {
      out[o++] = *in++;
      continue;
    }

    if ( in + 1 >= end )
      break;

    uint8_t count = in[1];
    in += 2;

    while ( count && o < row_bytes )
    {
      out[o++] = 0;
      --count;
    }
  }
}

class cluster_visibility
{
public:
  explicit cluster_visibility( const bsp_file& bsp );

  std::span<const uint8_t> row( int32_t cluster ) const
  {
    return { bits_.data() + size_t( cluster ) * row_bytes_, row_bytes_ };
  }

  int32_t cluster_count = 0;

private:
  size_t row_bytes_ = 0;
  std::vector<uint8_t> bits_;
};

cluster_visibility::cluster_visibility( const bsp_file& bsp )
{
  const std::span<const std::byte> bytes = bsp.visibility;

  if ( bytes.size() < sizeof( int32_t ) )
    return;

  const uint8_t* data = reinterpret_cast<const uint8_t*>( bytes.data() );
  const uint8_t* end = data + bytes.size();
  int32_t count;
  std::memcpy( &count, data, sizeof( count ) );

  if ( count <= 0 || vis_count_size + size_t( count ) * vis_offsets_size > bytes.size() )
    return;

  cluster_count = count;
  row_bytes_ = ( size_t( count ) + 7 ) / 8;
  bits_.resize( size_t( count ) * row_bytes_ );

  for ( int32_t c = 0; c < count; ++c )
  {
    int32_t pvs_offset;
    std::memcpy(
        &pvs_offset, data + vis_count_size + size_t( c ) * vis_offsets_size, sizeof( pvs_offset ) );

    if ( pvs_offset < 0 || size_t( pvs_offset ) >= bytes.size() )
      continue;

    decompress_row( data + pvs_offset, end, bits_.data() + size_t( c ) * row_bytes_, row_bytes_ );
  }
}

struct leaf_tables
{
  std::vector<uint32_t> cluster_leaf_first;
  std::vector<uint32_t> cluster_leaves;
  std::vector<uint32_t> leaf_face_first;
  std::vector<uint32_t> leaf_faces;

  leaf_tables( const bsp_file& bsp, int32_t cluster_count );
};

leaf_tables::leaf_tables( const bsp_file& bsp, int32_t cluster_count )
{
  cluster_leaf_first.resize( size_t( cluster_count ) + 1 );
  leaf_face_first.resize( bsp.leaves.size() + 1 );

  for ( size_t i = 0; i < bsp.leaves.size(); ++i )
  {
    const dleaf_v1& leaf = bsp.leaves[i];
    if ( leaf.cluster >= 0 && leaf.cluster < cluster_count )
      ++cluster_leaf_first[size_t( leaf.cluster ) + 1];

    leaf_face_first[i + 1] = leaf_face_first[i] + leaf.num_leaf_faces;
  }

  std::inclusive_scan(
      cluster_leaf_first.begin(), cluster_leaf_first.end(), cluster_leaf_first.begin() );

  cluster_leaves.resize( cluster_leaf_first.back() );
  std::vector<uint32_t> cursor( cluster_leaf_first.begin(), cluster_leaf_first.end() - 1 );

  leaf_faces.resize( leaf_face_first.back() );

  for ( size_t i = 0; i < bsp.leaves.size(); ++i )
  {
    const dleaf_v1& leaf = bsp.leaves[i];
    if ( leaf.cluster >= 0 && leaf.cluster < cluster_count )
      cluster_leaves[cursor[size_t( leaf.cluster )]++] = uint32_t( i );

    for ( uint16_t f = 0; f < leaf.num_leaf_faces; ++f )
    {
      leaf_faces[leaf_face_first[i] + f] = bsp.leaf_faces[size_t( leaf.first_leaf_face ) + f];
    }
  }
}

void build_cluster_faces( const patch_tree& tree, const cluster_visibility& vis,
    const leaf_tables& leaves, transfer_tables& tables )
{
  const size_t clusters = size_t( vis.cluster_count );
  std::vector<std::vector<uint32_t>> per_cluster( clusters );
  tbb::enumerable_thread_specific<std::vector<uint32_t>> stamps;

  tbb::parallel_for( size_t( 0 ), clusters,
      [&]( size_t c )
      {
        std::vector<uint32_t>& stamp = stamps.local();
        if ( stamp.size() != tree.face_root.size() )
          stamp.assign( tree.face_root.size(), std::numeric_limits<uint32_t>::max() );

        std::vector<uint32_t>& out = per_cluster[c];
        const std::span<const uint8_t> row = vis.row( int32_t( c ) );

        // the visible clusters in ascending order
        for ( size_t byte = 0; byte < row.size(); ++byte )
        {
          for ( uint8_t bits = row[byte]; bits != 0; bits &= uint8_t( bits - 1 ) )
          {
            const int32_t other = int32_t( byte * 8 + size_t( std::countr_zero( bits ) ) );
            if ( other >= vis.cluster_count )
              break;

            for ( uint32_t li = leaves.cluster_leaf_first[other];
                li < leaves.cluster_leaf_first[other + 1]; ++li )
            {
              const uint32_t leaf = leaves.cluster_leaves[li];

              for ( uint32_t fi = leaves.leaf_face_first[leaf];
                  fi < leaves.leaf_face_first[leaf + 1]; ++fi )
              {
                const uint32_t face = leaves.leaf_faces[fi];
                if ( stamp[face] == uint32_t( c ) )
                  continue;

                stamp[face] = uint32_t( c );

                if ( tree.face_root[face] >= 0 )
                  out.push_back( face );
              }
            }
          }
        }
      } );

  tables.cluster_first.reserve( clusters + 1 );
  size_t total = 0;

  for ( const std::vector<uint32_t>& list : per_cluster )
    total += list.size();

  tables.cluster_faces.reserve( total );

  for ( const std::vector<uint32_t>& list : per_cluster )
  {
    tables.cluster_first.push_back( uint32_t( tables.cluster_faces.size() ) );
    tables.cluster_faces.insert( tables.cluster_faces.end(), list.begin(), list.end() );
  }

  tables.cluster_first.push_back( uint32_t( tables.cluster_faces.size() ) );
}

} // namespace

transfer_tables build_transfer_tables( const bsp_file& bsp, const patch_tree& tree )
{
  transfer_tables tables;

  const cluster_visibility vis( bsp );
  if ( vis.cluster_count == 0 )
  {
    log_warn( "no vis clusters, direct lighting only" );
    return tables;
  }

  const leaf_tables leaves( bsp, vis.cluster_count );

  for ( const uint32_t leaf : tree.leaves )
  {
    const patch_node& patch = tree.nodes[leaf];
    if ( !patch.sky && patch.cluster >= 0 && patch.cluster < vis.cluster_count )
      tables.receivers.push_back( leaf );
  }

  build_cluster_faces( tree, vis, leaves, tables );
  return tables;
}
