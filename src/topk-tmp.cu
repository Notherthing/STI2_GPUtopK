#include "topk.h"

typedef uint4 group_t;  // uint32_t

// cudaSetDevice(0);
// std::iota(s_indices.begin(), s_indices.end(), 0);

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

// std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK,
// s_indices.end(),
//                 [&scores](const int& a, const int& b) {
//                     if (scores[a] != scores[b]) {
//                         return scores[a] > scores[b];  // 按照分数降序排序
//                     }
//                     return a < b;  // 如果分数相同，按索引从小到大排序
//             });

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
  // system_clock::time_point t0;
  // system_clock::time_point t1;
  // system_clock::time_point t2;
  // system_clock::time_point t3;
  // system_clock::time_point t4;
  // system_clock::time_point t5;
  // system_clock::time_point t6;
  // system_clock::time_point t7;
  // system_clock::time_point t8;
  // t2 = system_clock::now();
  // t0 = system_clock::now();

  cudaSetDevice(0);
  auto n_docs = docs.size();
  printf("n_docs:%lu \n", n_docs);
  // std::vector<float> scores(n_docs);
  // std::vector<int> s_indices(n_docs);
  // // cudaSetDevice(0);
  // std::iota(s_indices.begin(), s_indices.end(), 0);

  uint16_t *d_docs = nullptr;
  //  *d_query = nullptr;
  int *d_doc_lens = nullptr;

  // copy to device
  // cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  // cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);

  // uint16_t *h_docs;
  // t7 = system_clock::now();
  // cudaMallocHost(&h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  // t8 = system_clock::now();
  // std::cout<<"malloc host:"<<duration_cast<milliseconds>(t8 - t7).count()
  //           << "ms" << std::endl;
  // memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  // int *h_doc_lens_vec;
  // cudaMallocHost(&h_doc_lens_vec, sizeof(int) * n_docs);
  uint16_t *h_docs = new uint16_t[MAX_DOC_SIZE * n_docs];
  // memset(h_docs, 0, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  std::vector<int> h_doc_lens_vec(n_docs);
  // t1 = system_clock::now();
  // t0 = system_clock::now();
  // auto group_sz = sizeof(group_t) / sizeof(uint16_t);
  // auto layer_0_stride = n_docs * group_sz;
  // auto layer_1_stride = group_sz;

  std::vector<std::thread> threads;
  int docs_per_thread = n_docs / NUM_THREADS;
  for (int t = 0; t < NUM_THREADS; ++t) {
    int start_index = t * docs_per_thread;
    int end_index =
        (t == NUM_THREADS - 1) ? n_docs : start_index + docs_per_thread;
    threads.emplace_back([&, start_index, end_index]() {
      for (int i = start_index; i < end_index; i++) {
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
    });
  }

  // 等待所有线程完成
  for (auto &thread : threads) {
    thread.join();
  }
  // t2 = system_clock::now();
  // std::cout << "2 lop:" << duration_cast<milliseconds>(t1 - t0).count() <<
  // "ms"
  //           << std::endl;
  // cudaStream_t st_1, st_2;
  // cudaStreamCreate(&st_1);
  // cudaStreamCreate(&st_2);
  // cudaMallocAsync(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs, st_1);
  // cudaMallocAsync(&d_doc_lens, sizeof(int) * n_docs, st_2);
  // cudaMemcpyAsync(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
  //                 cudaMemcpyHostToDevice, st_1);
  // cudaMemcpyAsync(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs,
  //                 cudaMemcpyHostToDevice, st_2);

  cudaMalloc(&d_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs);
  cudaMalloc(&d_doc_lens, sizeof(int) * n_docs);
  // t2 = system_clock::now();
  cudaMemcpy(d_docs, h_docs, sizeof(uint16_t) * MAX_DOC_SIZE * n_docs,
             cudaMemcpyHostToDevice);
  // t3 = system_clock::now();
  cudaMemcpy(d_doc_lens, h_doc_lens_vec.data(), sizeof(int) * n_docs,
             cudaMemcpyHostToDevice);

  // cudaDeviceProp device_props;
  // cudaGetDeviceProperties(&device_props, 0);
  // t3 = system_clock::now();
  std::vector<int> s_indices(n_docs);
  // // cudaSetDevice(0);
  std::iota(s_indices.begin(), s_indices.end(), 0);
  // t4 = system_clock::now();
  // constexpr auto s_indices = generate_array<N>();
  // t3 = system_clock::now();
  // std::cout << "doc cpy:" << duration_cast<milliseconds>(t3 - t2).count()
  //           << "ms" << std::endl;
  // thrust::device_vector<int> dev_rank(n_docs);
  std::vector<int> s_ans(TOPK);
  // cudaMalloc(&d_query, sizeof(uint16_t) * MAX_DOC_SIZE);

  // t0 = system_clock::now();
  int query_num = querys.size();
  std::vector<float *> query_d_scores(query_num);
  std::vector<thrust::device_vector<int>> query_ranks(
      query_num, thrust::device_vector<int>(n_docs));
  std::vector<uint16_t *> query_data(query_num);
  std::vector<cudaStream_t> query_streams(query_num);
  // cudaDeviceSynchronize();
  // t4 = system_clock::now();
  for (int query_id = 0; query_id < query_num; ++query_id) {
    cudaMalloc(&query_d_scores[query_id], sizeof(float) * n_docs);
    // cudaMalloc(&query_ranks[query_id],sizeof(int) * n_docs);
    cudaMalloc(&query_data[query_id], sizeof(uint16_t) * MAX_DOC_SIZE);
    cudaStreamCreate(&query_streams[query_id]);
  }
  // t4 = system_clock::now();
  for (int query_id = 0; query_id < query_num; ++query_id) {
    cudaMemcpyAsync(thrust::raw_pointer_cast(query_ranks[query_id].data()),
                    s_indices.data(), n_docs * sizeof(int),
                    cudaMemcpyHostToDevice, query_streams[query_id]);
    const size_t query_len = querys[query_id].size();
    cudaMemcpyAsync(query_data[query_id], querys[query_id].data(),
                    sizeof(uint16_t) * query_len, cudaMemcpyHostToDevice,
                    query_streams[query_id]);
    // launch kernel
    int block = N_THREADS_IN_ONE_BLOCK;
    int grid = (n_docs + block - 1) / block;
    docQueryScoringCoalescedMemoryAccessSampleKernel<<<
        grid, block, 0, query_streams[query_id]>>>(
        d_docs, d_doc_lens, n_docs, query_data[query_id], query_len,
        query_d_scores[query_id]);
  }
  // printf("fine\n");
  // t5 = system_clock::now();
  // std::vector<std::vector<int>> rank_res(query_num, std::vector<int>(TOPK));

  for (int query_id = 0; query_id < query_num; ++query_id) {
    cudaStreamSynchronize(query_streams[query_id]);
    // printf("fine\n");
    thrust::sort(query_ranks[query_id].begin(), query_ranks[query_id].end(),
                 cmp(query_d_scores[query_id]));
    // cudaMemcpy(rank_res[query_id].data(),query_ranks[query_id],sizeof(int) *
    // TOPK,cudaMemcpyDeviceToHost);
    for (size_t i = 0; i < TOPK; ++i) {
      s_ans[i] = query_ranks[query_id][i];
    }
    // std::cout<<"fine"<<std::endl;
    indices.push_back(s_ans);
  }
  // t6 = system_clock::now();
  //   t1 = system_clock::now();
  // std::cout << "get res:" << duration_cast<milliseconds>(t1 - t0).count() <<
  // "ms"
  //           << std::endl;

  // for (auto &query : querys) {
  //   // init indices
  //   //  for (int i = 0; i < n_docs; ++i) {
  //   //      s_indices[i] = i;
  //   //  }
  //   // std::iota(s_indices.begin(), s_indices.end(), 0);
  //   // thrust::device_vector<int> dev_rank(n_docs);
  //   cudaMemcpy(thrust::raw_pointer_cast(dev_rank.data()), s_indices.data(),
  //              n_docs * sizeof(int), cudaMemcpyHostToDevice);

  //   const size_t query_len = query.size();
  //   // cudaMalloc(&d_query, sizeof(uint16_t) * query_len);
  //   cudaMemcpy(d_query, query.data(), sizeof(uint16_t) * query_len,
  //              cudaMemcpyHostToDevice);

  //   // launch kernel
  //   int block = N_THREADS_IN_ONE_BLOCK;
  //   int grid = (n_docs + block - 1) / block;
  //   docQueryScoringCoalescedMemoryAccessSampleKernel<<<grid, block>>>(
  //       d_docs, d_doc_lens, n_docs, d_query, query_len, d_scores);
  //   cudaDeviceSynchronize();

  //   // cudaMemcpy(scores.data(), d_scores, sizeof(float) * n_docs,
  //   //            cudaMemcpyDeviceToHost);

  //   // sort scores
  //   // std::partial_sort(s_indices.begin(), s_indices.begin() + TOPK,
  //   // s_indices.end(),
  //   //                 [&scores](const int& a, const int& b) {
  //   //                     if (scores[a] != scores[b]) {
  //   //                         return scores[a] > scores[b];  //
  //   //                         按照分数降序排序
  //   //                     }
  //   //                     return a < b;  // 如果分数相同，按索引从小到大排序
  //   //             });
  //   // t0 = system_clock::now();
  //   thrust::sort(dev_rank.begin(), dev_rank.end(), cmp(d_scores));
  //   // t1 = system_clock::now();
  //   // std::cout << "sort:" << duration_cast<milliseconds>(t1 - t0).count()
  //   <<
  //   // "ms"
  //   //           << std::endl;
  //   // thrust::host_vector<int> topk_rank(dev_rank.begin(),
  //   //                                    dev_rank.begin() + TOPK);

  //   // 这里有问题
  //   // std::vector<int> s_ans(TOPK);
  //   for (size_t i = 0; i < TOPK; ++i) {
  //     s_ans[i] = dev_rank[i];
  //   }
  //   // std::cout<<"fine"<<std::endl;
  //   indices.push_back(s_ans);

  //   // cudaFree(d_query);
  // }
  // cudaFree(d_query);
  // std::cout << "t1 - t0: " << duration_cast<milliseconds>(t1 - t0).count()
  //           << "ms" << std::endl;
  // std::cout << "t2 - t1: " << duration_cast<milliseconds>(t2 - t1).count()
  //           << "ms" << std::endl;
  // std::cout << "t3 - t2: " << duration_cast<milliseconds>(t3 - t2).count()
  //           << "ms" << std::endl;
  // std::cout << "t4 - t3: " << duration_cast<milliseconds>(t4 - t3).count()
  //           << "ms" << std::endl;
  // std::cout << "t5 - t4: " << duration_cast<milliseconds>(t5 - t4).count()
  //           << "ms" << std::endl;
  // std::cout << "t6 - t5: " << duration_cast<milliseconds>(t6 - t5).count()
  //           << "ms" << std::endl;
  // deallocation
  // for (int query_id = 0; query_id < query_num; ++query_id) {
  //   cudaFree(query_d_scores[query_id]);
  //   // cudaMalloc(&query_ranks[query_id],sizeof(int) * n_docs);
  //   cudaFree(query_data[query_id]);
  // }
  // cudaFree(d_docs);
  // // cudaFree(d_query);
  // // cudaFree(d_scores);
  // cudaFree(d_doc_lens);
  // cudaFreeHost(h_docs);
  // cud(h_docs);
}
