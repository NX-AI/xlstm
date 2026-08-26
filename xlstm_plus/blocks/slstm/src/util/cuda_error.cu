// Copyright (c) NXAI GmbH and its affiliates 2023
// Korbinian Poeppel

#include "cuda_error.h"

#define CEIL_DIV(a, b) (((a) + (b)-1) / (b))
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

void cudaOccupancyMaxActiveBlocksPerMultiprocessor2(dim3 blockSize,

                                                    size_t dynamicSMemSize,
                                                    const void *func) {
  // Obtain the current device properties
  int device;
  cudaFuncAttributes attr;
  cudaFuncGetAttributes(&attr, func);
  cudaGetDevice(&device);
  cudaDeviceProp deviceProperties;
  cudaGetDeviceProperties(&deviceProperties, device);

  // Estimate maximum number of warps and blocks per SM
  int maxWarpsPerSM =
      deviceProperties.maxThreadsPerMultiProcessor / deviceProperties.warpSize;
  int maxBlocksPerSM =
      deviceProperties.maxThreadsPerMultiProcessor / blockSize.x;

  // Calculate the number of warps required per block
  int warpsPerBlock = (blockSize.x - 1) / deviceProperties.warpSize + 1;

  // Blocks limited by warp count
  int blocksPerSM_WarpLimit = maxWarpsPerSM / warpsPerBlock;

  // Blocks limited by shared memory
  int blocksPerSM_SMemLimit =
      deviceProperties.sharedMemPerBlock / dynamicSMemSize;

  // Blocks limited by register usage
  int maxRegistersPerSM = deviceProperties.regsPerBlock;
  int registersNeededPerBlock = attr.numRegs * blockSize.x;
  int blocksPerSM_RegisterLimit = maxRegistersPerSM / registersNeededPerBlock;

  // Determine the limiting factor
  int blocksPerSM = MIN(MIN(blocksPerSM_WarpLimit, blocksPerSM_SMemLimit),
                        blocksPerSM_RegisterLimit);

  // Output the values
  printf(
      "Max Blocks per SM: %d, Max Warps per SM: %d, Warps per Block: %d, "
      "Blocks per SM (Warp Limit): %d, Blocks per SM (Shared Memory Limit): "
      "%d, Blocks per SM (Register Limit using %d regs): %d, Max Threads Per "
      "Block: %d, Local Size per Block: %lu, Blocks per SM: %d\n",
      maxBlocksPerSM, maxWarpsPerSM, warpsPerBlock, blocksPerSM_WarpLimit,
      blocksPerSM_SMemLimit, attr.numRegs, blocksPerSM_RegisterLimit,
      attr.maxThreadsPerBlock, attr.localSizeBytes, blocksPerSM);
}