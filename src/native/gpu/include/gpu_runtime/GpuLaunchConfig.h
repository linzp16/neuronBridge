#pragma once

#include <cuda_runtime.h>

namespace npgr {

constexpr int kNeuronLaunchBlocksPerSm = 8;
constexpr int kLinearScanBlocksPerSm = 8;
constexpr int kWorkItemLaunchBlocksPerSm = 4;
constexpr int kHistoryScanBlocksPerSm = 4;

struct LaunchConfig1D {
    int threads;
    int blocks;
};

inline int CeilDivInt(int n, int d) {
    return (n + d - 1) / d;
}

inline int QuerySmCount() {
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return 1;
    }
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
        return 1;
    }
    return prop.multiProcessorCount > 0 ? prop.multiProcessorCount : 1;
}

inline int QueryMaxBlocksFromSmCount(int blocks_per_sm) {
    const int sm_count = QuerySmCount();
    return sm_count * (blocks_per_sm > 0 ? blocks_per_sm : 1);
}

inline int MaxBlocksFromSmCount(int sm_count, int blocks_per_sm) {
    const int normalized_sm_count = sm_count > 0 ? sm_count : 1;
    return normalized_sm_count * (blocks_per_sm > 0 ? blocks_per_sm : 1);
}

inline LaunchConfig1D MakeLaunchConfig1DWithMaxBlocks(int item_count, int threads, int max_blocks) {
    LaunchConfig1D cfg;
    cfg.threads = threads > 0 ? threads : 256;
    const int requested_blocks = item_count > 0 ? CeilDivInt(item_count, cfg.threads) : 1;
    const int normalized_max_blocks = max_blocks > 0 ? max_blocks : 1;
    cfg.blocks = requested_blocks < normalized_max_blocks ? requested_blocks : normalized_max_blocks;
    if (cfg.blocks <= 0) {
        cfg.blocks = 1;
    }
    return cfg;
}

inline LaunchConfig1D MakeLaunchConfig1D(int item_count, int threads, int blocks_per_sm) {
    return MakeLaunchConfig1DWithMaxBlocks(
        item_count,
        threads,
        QueryMaxBlocksFromSmCount(blocks_per_sm));
}

inline int MakeCappedGridXWithMaxBlocks(int logical_block_count, int max_blocks) {
    const int requested = logical_block_count > 0 ? logical_block_count : 1;
    const int normalized_max_blocks = max_blocks > 0 ? max_blocks : 1;
    return requested < normalized_max_blocks ? requested : normalized_max_blocks;
}

inline int MakeCappedGridX(int logical_block_count, int blocks_per_sm) {
    return MakeCappedGridXWithMaxBlocks(
        logical_block_count,
        QueryMaxBlocksFromSmCount(blocks_per_sm));
}

}  // namespace npgr
