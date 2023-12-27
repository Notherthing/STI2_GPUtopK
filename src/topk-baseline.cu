
#include "topk.h"

typedef uint4 group_t;  // uint32_t
using namespace std;
using namespace std::chrono;

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const int *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores) {
  // each thread process one doc-query pair scoring task
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x,
                tnum = gridDim.x * blockDim.x;

  if (tid >= n_docs) {
    return;
  }

  __shared__ uint16_t query_on_shm[MAX_QUERY_SIZE];
#pragma unroll
  for (auto i = threadIdx.x; i < query_len; i += blockDim.x) {
    query_on_shm[i] = query[i];  // not very efficient query loading temporally,
                                 // as assuming its not hotspot
  }

  __syncthreads();

  for (auto doc_id = tid; doc_id < n_docs; doc_id += tnum) {
    register int query_idx = 0;

    register float tmp_score = 0.;

    register bool no_more_load = false;

    for (auto i = 0; i < MAX_DOC_SIZE / (sizeof(group_t) / sizeof(uint16_t));
         i++) {
      if (no_more_load) {
        break;
      }
      register group_t loaded = ((group_t *)docs)[i * n_docs + doc_id];  // tid
      register uint16_t *doc_segment = (uint16_t *)(&loaded);
      for (auto j = 0; j < sizeof(group_t) / sizeof(uint16_t); j++) {
        if (doc_segment[j] == 0) {
          no_more_load = true;
          break;
          // return;
        }
        while (query_idx < query_len &&
               query_on_shm[query_idx] < doc_segment[j]) {
          ++query_idx;
        }
        if (query_idx < query_len) {
          tmp_score += (query_on_shm[query_idx] == doc_segment[j]);
        }
      }
      __syncwarp();
    }
    scores[doc_id] = tmp_score / max(query_len, doc_lens[doc_id]);  // tid
  }
}

void doc_query_scoring_gpu_function(
    std::vector<std::vector<uint16_t>> &querys,
    std::vector<std::vector<uint16_t>> &docs, std::vector<uint16_t> &lens,
    std::vector<std::vector<int>> &indices  // shape [querys.size(), TOPK]
) {
  system_clock::time_point t0;
  system_clock::time_point t1;
  system_clock::time_point t2;
  system_clock::time_point t3;

  auto n_docs = docs.size();
  std::vector<float> scores(n_docs);
  std::vector<int> s_indices(n_docs);

  float *d_scores = nullptr;
  uint16_t *d_docs = nullptr, *d_query = nullptr;
  int *d_doc_lens = nullptr;

  // copy to device
  cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  cudaMalloc(&d_scores, sizeof(float) * n_docs);
  cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

  t0 = system_clock::now();
  uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
  t1 = system_clock::now();
  std::cout << "new memory:" << duration_cast<milliseconds>(t1 - t0).count()
            << "ms" << std::endl;
  uint16_t *h_docs_2;
  t0 = system_clock::now();
  cudaMallocHost(&h_docs_2, MAX_DOC_SIZE * n_docs * sizeof(uint16_t));
  t1 = system_clock::now();
  std::cout << "pinned memory:" << duration_cast<milliseconds>(t1 - t0).count()
            << "ms" << std::endl;
  std::cout<<"total size:" << (MAX_DOC_SIZE * n_docs * sizeof(uint16_t) /(1024*1024))<<std::endl;
  // memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  std::vector<int> h_doc_lens_vec(n_docs);
  t0 = system_clock::now();
  for (int i = 0; i < docs.size(); i++) {
    for (int j = 0; j < docs[i].size(); j++) {
      auto group_sz = sizeof(group_t) / sizeof(uint16_t);
      auto layer_0_offset = j / group_sz;
      auto layer_0_stride = n_docs * group_sz;
      auto layer_1_offset = i;
      auto layer_1_stride = group_sz;
      auto layer_2_offset = j % group_sz;
      auto final_offset = layer_0_offset * layer_0_stride +
                          layer_1_offset * layer_1_stride + layer_2_offset;
      h_docs[final_offset] = docs[i][j];
    }
    h_doc_lens_vec[i] = docs[i].size();
  }
  t1 = system_clock::now();
  std::cout << "process:"
            << duration_cast<milliseconds>(t1 - t0).count() << "ms"
            << std::endl;

  t0 = system_clock::now();
  for (int i = 0; i < docs.size(); i++) {
    for (int j = 0; j < docs[i].size(); j++) {
      auto group_sz = sizeof(group_t) / sizeof(uint16_t);
      auto layer_0_offset = j / group_sz;
      auto layer_0_stride = n_docs * group_sz;
      auto layer_1_offset = i;
      auto layer_1_stride = group_sz;
      auto layer_2_offset = j % group_sz;
      auto final_offset = layer_0_offset * layer_0_stride +
                          layer_1_offset * layer_1_stride + layer_2_offset;
      h_docs_2[final_offset] = docs[i][j];
    }
    h_doc_lens_vec[i] = docs[i].size();
  }
  t1 = system_clock::now();
  std::cout << "process_pin:"
            << duration_cast<milliseconds>(t1 - t0).count() << "ms"
            << std::endl;

  t0 = system_clock::now();
  cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
             cudaMemcpyHostToDevice);
  t1 = system_clock::now();
  std::cout << "pageable trans memory:"
            << duration_cast<milliseconds>(t1 - t0).count() << "ms"
            << std::endl;
  t0 = system_clock::now();
  cudaMemcpy(d_docs, h_docs_2, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
             cudaMemcpyHostToDevice);
  t1 = system_clock::now();
  std::cout << "pinned trans memory:"
            << duration_cast<milliseconds>(t1 - t0).count() << "ms"
            << std::endl;
  cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs,
             cudaMemcpyHostToDevice);
  cudaDeviceProp device_props;
  cudaGetDeviceProperties(&device_props, 0);

  cudaSetDevice(0);

  for (auto &query : querys) {
    // init indices
    for (int i = 0; i < n_docs; ++i) {
      s_indices[i] = i;
    }

    const size_t query_len = query.size();
    cudaMalloc(&d_query, sizeof(uint16_t) * query_len);
    cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len,
               cudaMemcpyHostToDevice);

    // launch kernel
    int block = N_THREADS_IN_ONE_BLOCK;
    int grid = (n_docs + block - 1) / block;
    docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block>>>(
        d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores);
    cudaDeviceSynchronize();

    cudaMemcpy(scores.data(), d_scores, sizeof(float) * n_docs,
               cudaMemcpyDeviceToHost);

    // t0 = system_clock::now();
    // sort scores
    std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK,
                      s_indices.end(), [&scores](const int &a, const int &b) {
                        if (scores[a] != scores[b]) {
                          return scores[a] > scores[b];  // 按照分数降序排序
                        }
                        return a < b;  // 如果分数相同，按索引从小到大排序
                      });
    // t1 = system_clock::now();
    // std::cout << "sort:" << duration_cast<milliseconds>(t1 - t0).count() <<
    // "ms"
    //           << std::endl;
    std::vector<int> s_ans(s_indices.begin(), s_indices.begin() + TOPK);
    indices.push_back(s_ans);

    cudaFree(d_query);
  }

  // deallocation
  cudaFree(d_docs);
  // cudaFree(d_query);
  cudaFree(d_scores);
  cudaFree(d_doc_lens);
  free(h_docs);
}
