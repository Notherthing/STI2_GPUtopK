#include "topk.h"

typedef uint4 group_t;  // uint32_t

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const uint8_t *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores) {
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
  uint16_t docs_chunk[4];
  // Using vectorization to process multiple elements at once
  #pragma unroll
  for (uint8_t j = 0; j < doc_len; j += 4) {

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
    if (j + k < doc_len) {
        docs_chunk[k] = docs[tid * MAX_DOC_SIZE + j + k];
      } else {
        docs_chunk[k] = 0;  // Padding with zeros for elements beyond array bounds
      }
    }

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
      while (query_idx < query_len && query_on_shm[query_idx] < docs_chunk[k]) {
        ++query_idx;
      }

      if (query_idx < query_len) {
        tmp_score += (query_on_shm[query_idx] == docs_chunk[k]);
      }
    }
  }

  scores[tid] = float(tmp_score) / max(query_len, doc_len);
}

struct cmp {
  cmp(const float *distances_) : distances(distances_) {}
  const float *distances;
  __thrust_exec_check_disable__ __host__ __device__ bool operator()(
      const int &lhs, const int &rhs) const {
    if (distances[lhs] != distances[rhs]) {
      return distances[lhs] > distances[rhs];
    }
    return lhs < rhs;
  }
};  // end less

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices  // shape [querys.size(), TOPK]
) {

  // t0 = system_clock::now();
  // init
  cudaSetDevice(0);
  size_t n_docs = docs.size();
  int query_num = querys.size();
  assert(query_num <350);
  std::vector<uint16_t *> doc_data_per_batch(BATCH_NUM, nullptr);
  std::vector<uint8_t *> doc_len_per_batch(BATCH_NUM, nullptr);
  std::vector<uint16_t *> d_doc_data_per_batch(BATCH_NUM, nullptr);
  std::vector<uint8_t *> d_doc_len_per_batch(BATCH_NUM, nullptr);
  std::vector<bool> h_batch_alloc(BATCH_NUM, false);
  std::vector<bool> d_batch_alloc(BATCH_NUM, false);
  std::vector<cudaStream_t> batch_streams(BATCH_NUM);
  std::vector<std::thread> threads;
  std::vector<float *> query_d_scores(QUERY_ONE_TERM, nullptr);
  std::vector<uint16_t *> query_data(query_num, nullptr);
  std::vector<thrust::device_vector<int>> query_ranks(
      2, thrust::device_vector<int>(n_docs));

  std::vector<int> s_ans(TOPK);
  indices.reserve(query_num);
  int *s_indices = new int[n_docs];
  // cudaMallocHost(&s_indices, sizeof(int) * n_docs);
  std::iota(s_indices, s_indices + n_docs, 0);
  cudaMemcpyAsync(thrust::raw_pointer_cast(query_ranks[0].data()), s_indices,
             n_docs * sizeof(int), cudaMemcpyHostToDevice);
  // int *s_indices = nullptr;

  threads.reserve(BATCH_NUM);
  for (int i = 0; i < BATCH_NUM; ++i) {
    cudaStreamCreate(&batch_streams[i]);
  }
  // 显存占用很大
  // float *query_scores_d = nullptr;
  // cudaMalloc(&query_scores_d, sizeof(float) * n_docs);
  for (int i = 0; i < query_num; ++i) {
    // cudaMalloc(&query_d_scores[i], sizeof(float) * n_docs);
    cudaMalloc(&query_data[i], sizeof(uint16_t) * MAX_DOC_SIZE);
    cudaMemcpy(query_data[i], querys[i].data(),
               sizeof(uint16_t) * querys[i].size(), cudaMemcpyHostToDevice);
  }
  for (int i = 0; i < QUERY_ONE_TERM; ++i) {
    cudaMalloc(&query_d_scores[i], sizeof(float) * n_docs);
  }

  for (size_t query_done = 0; query_done < query_num;) {
    size_t this_batch_n_ques = ((query_done + QUERY_ONE_TERM) >= query_num)
                                   ? query_num - query_done
                                   : QUERY_ONE_TERM;
    // cudaMemset(query_scores_d, 0, sizeof(float) * n_docs);
    size_t done_docs_n = 0;
    int batch_id = 0;
    for (done_docs_n = 0; done_docs_n < n_docs;) {
      // printf("done_docs: %lu\n", done_docs_n);
      batch_id = batch_id % BATCH_NUM;
      while (cudaStreamQuery(batch_streams[batch_id]) != cudaSuccess) {
        ++batch_id;
        batch_id = batch_id % BATCH_NUM;
      }
      // printf("fine1\n");
      size_t this_batch_n_docs = ((done_docs_n + BATCH_SIZE) >= n_docs)
                                     ? n_docs - done_docs_n
                                     : BATCH_SIZE;
      // printf("this_batch_n_docs: %lu\n", this_batch_n_docs);
      if (h_batch_alloc[batch_id] == false) {
        cudaHostAlloc(&doc_data_per_batch[batch_id],
                      sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE,
                      cudaHostAllocDefault);
        cudaHostAlloc(&doc_len_per_batch[batch_id], sizeof(uint8_t) * BATCH_SIZE,
                      cudaHostAllocDefault);

        h_batch_alloc[batch_id] = true;
      }

      if (d_batch_alloc[batch_id] == false) {
        cudaMalloc(&d_doc_data_per_batch[batch_id],
                   sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE);
        cudaMalloc(&d_doc_len_per_batch[batch_id], sizeof(uint8_t) * BATCH_SIZE);
        d_batch_alloc[batch_id] = true;
      }
      // printf("fine2\n");
      // 多线程置位,这里没问题
      threads.clear();
      // memset(doc_data_per_batch[batch_id],0,sizeof(uint16_t) * MAX_DOC_SIZE *
      // BATCH_SIZE); memset(doc_len_per_batch[batch_id], 0,sizeof(int) *
      // BATCH_SIZE);
      int docs_per_thread = this_batch_n_docs / NUM_THREADS;
      for (int t = 0; t < NUM_THREADS; ++t) {
        int start_index = t * docs_per_thread;
        int end_index = (t == NUM_THREADS - 1) ? this_batch_n_docs
                                               : start_index + docs_per_thread;
        threads.emplace_back([&, start_index, end_index, done_docs_n]() {
          // size_t doc_sz = 0;
          for (int i = start_index; i < end_index; i++) {
            // for (int j = 0; j < docs[i + done_docs_n].size(); j++) {
            //   auto final_offset = i * MAX_DOC_SIZE + j;
            //   doc_data_per_batch[batch_id][final_offset] =
            //       docs[i + done_docs_n][j];
            // }
            doc_len_per_batch[batch_id][i] = docs[i + done_docs_n].size() ;
            memcpy(doc_data_per_batch[batch_id] + i * MAX_DOC_SIZE,
                   docs[i + done_docs_n].data(),
                   doc_len_per_batch[batch_id][i] * sizeof(uint16_t));
            // doc_len_per_batch[batch_id][i] = docs[i + done_docs_n].size() ;
          }
        });
      }

      for (auto &thread : threads) {
        thread.join();
      }
      // printf("fine3\n");
      cudaMemcpyAsync(d_doc_data_per_batch[batch_id],
                      doc_data_per_batch[batch_id],
                      sizeof(uint16_t) * MAX_DOC_SIZE * this_batch_n_docs,
                      cudaMemcpyHostToDevice, batch_streams[batch_id]);
      cudaMemcpyAsync(d_doc_len_per_batch[batch_id],
                      doc_len_per_batch[batch_id],
                      sizeof(uint8_t) * this_batch_n_docs, cudaMemcpyHostToDevice,
                      batch_streams[batch_id]);

      int block = N_THREADS_IN_ONE_BLOCK;
      int grid = (this_batch_n_docs + block - 1) / block;
      // printf("fine4\n");
      for (int query_id = 0; query_id < this_batch_n_ques; ++query_id) {
        docQueryScoringCoalescedMemoryAccessSampleKernel<<<
            grid, block, 0, batch_streams[batch_id]>>>(
            d_doc_data_per_batch[batch_id], d_doc_len_per_batch[batch_id],
            this_batch_n_docs, query_data[query_id + query_done], querys[query_id + query_done].size(),
            query_d_scores[query_id] + done_docs_n);
      }
      ++batch_id;
      done_docs_n += this_batch_n_docs;
    }
    int last_batch_id =
        batch_id == -1 ? (BATCH_NUM - 1) : (batch_id - 1) % BATCH_NUM;
      // printf("fine\n");

    cudaStreamSynchronize(batch_streams[last_batch_id]);
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
    query_done += QUERY_ONE_TERM;
    // printf("done query:%d\n",query_done);
  }
  // system_clock::time_point t0;
  // system_clock::time_point t1;
  // // t0 = system_clock::now();
  // // // cudaStreamQuery(batch_streams[0]);
  // // cudaLaunchHostFunc(batch_streams[0],MyCallback,NULL);
  // // t1 = system_clock::now();
  //   std::cout << "t1 - t0: " << duration_cast<nanoseconds>(t1 - t0).count()
  //           << "ns" << std::endl;
}