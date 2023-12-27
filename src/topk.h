#pragma once
#include <bits/stdc++.h>
#include <cuda.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

//
#define MAX_DOC_SIZE 128
#define MAX_QUERY_SIZE 4096
#define N_THREADS_IN_ONE_BLOCK 256
#define TOPK 100

// the number of CPU threads for preprocessing docs data to pinned memory.
#define NUM_THREADS 8
// the number of documents processed per batch computation.
#define BATCH_SIZE 25600
// the number of internal CUDA streams, with each stream executing one batch.
#define BATCH_NUM 36
// the number of queries for top-K search executed per term.
#define QUERY_ONE_TERM 80
// the loop size for loop unroll inside the cuda kernel function.
#define ROLL_INNER 4
// bitset size 1568 * 32 > 50000
#define MAX_ELEMENT 1568


void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>>& query,
                                    std::vector<std::vector<uint16_t>>& docs,
                                    std::vector<uint16_t>& lens,
                                    std::vector<std::vector<int>>& indices);

/**
 * @brief Set the CPU Id for thread.
 * Bind each thread to a specific CPU core to prevent thread contention during
 * data preprocessing.
 *
 * @param th the thread
 * @param cpuid the cpu ID
 * @return true- set cpu id fine.
 * @return false- set cpu id failed.
 */
bool setCpuId(std::thread& th, int8_t cpuid);