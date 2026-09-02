#pragma once

#include "bsp.h"
#include "patches.h"

#include <vector>

// the receivers and the faces their clusters can see
struct transfer_tables
{
  // non sky leaves with a cluster, in node order
  std::vector<uint32_t> receivers;
  // the faces with patches in each cluster's pvs
  std::vector<uint32_t> cluster_first;
  std::vector<uint32_t> cluster_faces;
};

transfer_tables build_transfer_tables( const bsp_file& bsp, const patch_tree& tree );
