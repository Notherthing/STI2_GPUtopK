#include "topk.h"

typedef uint4 group_t;  // uint32_t

bool setCpuId(std::thread& th, int8_t cpuid) {
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(cpuid, &cpu_set);
  if (!pthread_setaffinity_np(th.native_handle(), sizeof(cpu_set_t),
                              &cpu_set)) {
    return true;
  }
  std::cerr << "Failed to set Thread cpu setaffinity :" << std::strerror(errno)
            << std::endl;
  return false;
}

// Inline max function.
__device__ __forceinline__ int inlineMax(int a, int b) {
  return (a > b) ? a : b;
}

/**
 * @brief  computing the vector intersection between a specific query and
 * documents.
 *
 * @param docs the pointer to the docs.
 * @param doc_lens the pointer to the length array of docs.
 * @param n_docs the total num of docs.
 * @param query the pointer to the query vector.
 * @param query_len the length of the query vector.
 * @param scores the result of scores.
 */
void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t* docs,
    const uint8_t* doc_lens,
    const size_t n_docs,
    uint16_t* query,
    const int query_len,
    int16_t* scores) {
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= n_docs) {
    return;
  }
    __shared__ uint16_t query_on_shm[MAX_DOC_SIZE];
  #pragma unroll
    for (auto i = 0; i < query_len; ++i) {
      query_on_shm[i] = query[i];
    }
  __syncthreads();
  register uint8_t query_idx = 0;
  register uint8_t doc_len = doc_lens[tid];
  register uint8_t tmp_score = 0;

    uint16_t docs_chunk[ROLL_INNER];
  #pragma unroll
    for (uint8_t j = 0; j < doc_len; j += ROLL_INNER) {
  #pragma unroll
      for (int k = 0; k < ROLL_INNER; ++k) {
        if (j + k < doc_len) {
          docs_chunk[k] = docs[tid * MAX_DOC_SIZE + j + k];
        } else {
          docs_chunk[k] =
              0;  // Padding with zeros for elements beyond array bounds
        }
      }
  #pragma unroll
      for (int k = 0; k < ROLL_INNER; ++k) {
        while (query_idx < query_len) {
          if (query_on_shm[query_idx] < docs_chunk[k]) {
            ++query_idx;
          } else {
            tmp_score += (query_on_shm[query_idx] == docs_chunk[k]);
            break;
          }
        }
      }
    }
  scores[tid] = static_cast<int16_t>((128 * 128 * float(tmp_score)) /
                                     inlineMax(query_len, doc_len));
}

/**
 * @brief Struct functor for determining the order of rank ID.
 * @param distances_ the pointer to the array 'distances_' used for
 * determining rank distances.
 */
struct cmp {
  cmp(const int16_t* distances_) : distances(distances_) {}
  const int16_t* distances;
  __thrust_exec_check_disable__ __host__ __device__ bool operator()(
      const int& lhs,
      const int& rhs) const {
    if (distances[lhs] != distances[rhs]) {
      return distances[lhs] > distances[rhs];
    }
    return lhs < rhs;
  }
};  // end less

void doc_query_scoring_gpu_function(std::vector<std::vector<uint16_t>>& querys,
                                    std::vector<std::vector<uint16_t>>& docs,
                                    std::vector<uint16_t>& lens,
                                    std::vector<std::vector<int>>& indices) {
  // init
  /*
  1. init device.
  2. get the nums of docs and queries.
  3. build the resources.
  */
  cudaSetDevice(0);
  size_t n_docs = docs.size();
  int query_num = querys.size();

  // host-pinned memory buffer for each batch docs data.
  std::vector<uint16_t*> doc_data_per_batch(BATCH_NUM, nullptr);
  // host-pinned memory buffer for each batch docs len.
  std::vector<uint8_t*> doc_len_per_batch(BATCH_NUM, nullptr);
  // device memory buffer for each batch docs data.
  std::vector<uint16_t*> d_doc_data_per_batch(BATCH_NUM, nullptr);
  // device memory for each batch docs len.
  std::vector<uint8_t*> d_doc_len_per_batch(BATCH_NUM, nullptr);
  // flags for checking if the host buffer for the current stream is allocated;
  // if not, allocate it on-the-fly.
  std::vector<bool> h_batch_alloc(BATCH_NUM, false);
  // flags for checking if the device memory for the current stream is
  // allocated; if not, allocate it on-the-fly.
  std::vector<bool> d_batch_alloc(BATCH_NUM, false);
  // cudaStreams for all batches.
  std::vector<cudaStream_t> batch_streams(BATCH_NUM);
  // threads for preprocessing.
  std::vector<std::thread> threads;
  // The score calculation results for each query, requiring space only for the
  // calculations of all queries in a single term at a time.
  // only in device memory.
  std::vector<int16_t*> query_d_scores(QUERY_ONE_TERM, nullptr);
  // device memory pointers for storing queries data.
  std::vector<uint16_t*> query_data(query_num, nullptr);
  // device arrays for storing ranks.
  std::vector<thrust::device_vector<int>> query_ranks(
      2, thrust::device_vector<int>(n_docs));

  /*
   * init the 0 - n_docs rank sequence.
   * And copy it to device.
   */
  std::vector<int> s_ans(TOPK);
  indices.reserve(query_num);
  int* s_indices = new int[n_docs];
  std::iota(s_indices, s_indices + n_docs, 0);
  cudaMemcpyAsync(thrust::raw_pointer_cast(query_ranks[0].data()), s_indices,
                  n_docs * sizeof(int), cudaMemcpyHostToDevice);

  /**
   * @brief init resources
   *  threads vector, cudaStreams vectors and data copy.
   */
  threads.reserve(BATCH_NUM);
  for (int i = 0; i < BATCH_NUM; ++i) {
    cudaStreamCreate(&batch_streams[i]);
  }

  for (int i = 0; i < query_num; ++i) {
    cudaMalloc(&query_data[i], sizeof(uint16_t) * MAX_DOC_SIZE);
    cudaMemcpy(query_data[i], querys[i].data(),
               sizeof(uint16_t) * querys[i].size(), cudaMemcpyHostToDevice);
  }
  for (int i = 0; i < QUERY_ONE_TERM; ++i) {
    cudaMalloc(&query_d_scores[i], sizeof(int16_t) * n_docs);
  }

  // start to calculate.
  for (size_t query_done = 0; query_done < query_num;) {
    size_t this_batch_n_ques = ((query_done + QUERY_ONE_TERM) >= query_num)
                                   ? query_num - query_done
                                   : QUERY_ONE_TERM;
    size_t done_docs_n = 0;
    int batch_id = 0;
    for (done_docs_n = 0; done_docs_n < n_docs;) {
      // select the batch(stream) id.
      batch_id = batch_id % BATCH_NUM;
      while (cudaStreamQuery(batch_streams[batch_id]) != cudaSuccess) {
        ++batch_id;
        batch_id = batch_id % BATCH_NUM;
      }

      // get this batch n docs.
      size_t this_batch_n_docs = ((done_docs_n + BATCH_SIZE) >= n_docs)
                                     ? n_docs - done_docs_n
                                     : BATCH_SIZE;

      // check the memory allocated or not.
      if (h_batch_alloc[batch_id] == false) {
        cudaHostAlloc(&doc_data_per_batch[batch_id],
                      sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE,
                      cudaHostAllocDefault);
        cudaHostAlloc(&doc_len_per_batch[batch_id],
                      sizeof(uint8_t) * BATCH_SIZE, cudaHostAllocDefault);

        h_batch_alloc[batch_id] = true;
      }

      if (d_batch_alloc[batch_id] == false) {
        cudaMalloc(&d_doc_data_per_batch[batch_id],
                   sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE);
        cudaMalloc(&d_doc_len_per_batch[batch_id],
                   sizeof(uint8_t) * BATCH_SIZE);
        d_batch_alloc[batch_id] = true;
      }

      // preprocess the docs data and lens use multi-threads.
      threads.clear();
      int docs_per_thread = this_batch_n_docs / NUM_THREADS;
      for (int t = 0; t < NUM_THREADS; ++t) {
        int start_index = t * docs_per_thread;
        int end_index = (t == NUM_THREADS - 1) ? this_batch_n_docs
                                               : start_index + docs_per_thread;
        threads.emplace_back([&, start_index, end_index, done_docs_n]() {
          for (int i = start_index; i < end_index; i++) {
            doc_len_per_batch[batch_id][i] = docs[i + done_docs_n].size();
            memcpy(doc_data_per_batch[batch_id] + i * MAX_DOC_SIZE,
                   docs[i + done_docs_n].data(),
                   doc_len_per_batch[batch_id][i] * sizeof(uint16_t));
          }
        });
        setCpuId(threads[t], t + 1);
      }

      for (auto& thread : threads) {
        thread.join();
      }

      // async copy the data and lens, the launch the cuda kernel.
      cudaMemcpyAsync(d_doc_data_per_batch[batch_id],
                      doc_data_per_batch[batch_id],
                      sizeof(uint16_t) * MAX_DOC_SIZE * this_batch_n_docs,
                      cudaMemcpyHostToDevice, batch_streams[batch_id]);
      cudaMemcpyAsync(d_doc_len_per_batch[batch_id],
                      doc_len_per_batch[batch_id],
                      sizeof(uint8_t) * this_batch_n_docs,
                      cudaMemcpyHostToDevice, batch_streams[batch_id]);

      int block = N_THREADS_IN_ONE_BLOCK;
      int grid = (this_batch_n_docs + block - 1) / block;
      for (int query_id = 0; query_id < this_batch_n_ques; ++query_id) {
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<
            grid, block, 0, batch_streams[batch_id]>>>(
            d_doc_data_per_batch[batch_id], d_doc_len_per_batch[batch_id],
            this_batch_n_docs, query_data[query_id + query_done],
            querys[query_id + query_done].size(),
            query_d_scores[query_id] + done_docs_n);
      }
      ++batch_id;
      done_docs_n += this_batch_n_docs;
    }

    // when the queries-docs' scores of the term finished.
    // get the top-K results.
    int last_batch_id =
        batch_id == -1 ? (BATCH_NUM - 1) : (batch_id - 1) % BATCH_NUM;
    // sync last batch
    cudaStreamSynchronize(batch_streams[last_batch_id]);

    // use device to device get 0-n_docs seq.
    // then use GPU do the sort.
    // get results.
    for (int query_id = 0; query_id < this_batch_n_ques; ++query_id) {
      cudaMemcpy(thrust::raw_pointer_cast(query_ranks[1].data()),
                 thrust::raw_pointer_cast(query_ranks[0].data()),
                 n_docs * sizeof(int), cudaMemcpyDeviceToDevice);
      thrust::sort(query_ranks[1].begin(), query_ranks[1].end(),
                   cmp(query_d_scores[query_id]));
      for (size_t i = 0; i < TOPK; ++i) {
        s_ans[i] = query_ranks[1][i];
      }
      indices.push_back(s_ans);
    }

    // record the finished queries' num of this term.
    query_done += QUERY_ONE_TERM;
  }
}