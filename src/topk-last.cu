#include "topk.h"

typedef uint4 group_t;  // uint32_t

int set_device() {
  cudaSetDevice(0);
  return 0;
}

static const int g_cuda = set_device();
std::vector<uint16_t *> doc_data_per_batch(BATCH_NUM, nullptr);
std::vector<int *> doc_len_per_batch(BATCH_NUM, nullptr);
std::vector<uint16_t *> d_doc_data_per_batch(BATCH_NUM, nullptr);
std::vector<int *> d_doc_len_per_batch(BATCH_NUM, nullptr);
std::vector<bool> h_batch_alloc(BATCH_NUM, false);
std::vector<bool> d_batch_alloc(BATCH_NUM, false);
std::vector<cudaStream_t> batch_streams(BATCH_NUM);
std::vector<std::thread> threads;
std::vector<int> s_ans(TOPK);
int *s_indices = nullptr;

int cuda_malloc_pre() {
  for (int batch_id = 0; batch_id < BATCH_NUM; ++batch_id) {
    cudaStreamCreate(&batch_streams[batch_id]);
    if (h_batch_alloc[batch_id] == false) {
      cudaMallocHost(&doc_data_per_batch[batch_id],
                     sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE);
      cudaMallocHost(&doc_len_per_batch[batch_id], sizeof(int) * BATCH_SIZE);
      h_batch_alloc[batch_id] = true;
    }
    // cudaStreamSynchronize(batch_streams[batch_id]);
    // if (d_doc_data_per_batch[batch_id] == nullptr) {
    //   cudaMallocAsync(&d_doc_data_per_batch[batch_id],
    //                   sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE,
    //                   batch_streams[batch_id]);
    //   cudaMallocAsync(&d_doc_len_per_batch[batch_id], sizeof(int) *
    //   BATCH_SIZE,
    //                   batch_streams[batch_id]);
    // }
    if (d_batch_alloc[batch_id] == false) {
      cudaMalloc(&d_doc_data_per_batch[batch_id],
                 sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE);
      cudaMalloc(&d_doc_len_per_batch[batch_id], sizeof(int) * BATCH_SIZE);
      d_batch_alloc[batch_id] = true;
    }
  }
  return 0;
}

static const int g_cuda_malloc = cuda_malloc_pre();

int rank_set_pre() {
  cudaMallocHost(&s_indices, sizeof(int) * MAX_N_DOCS);
  std::iota(s_indices, s_indices + MAX_N_DOCS, 0);
  return 0;
}

static const int g_rank_set = rank_set_pre();

void __global__ docQueryScoringCoalescedMemoryAccessSampleKernel(
    const __restrict__ uint16_t *docs, const int *doc_lens, const size_t n_docs,
    uint16_t *query, const int query_len, float *scores) {
  // each thread process one doc-query pair scoring task
  register auto tid = blockIdx.x * blockDim.x + threadIdx.x;

  if (tid >= n_docs) {
    return;
  }
  __shared__ uint16_t query_on_shm[MAX_DOC_SIZE];
#pragma unroll
  for (auto i = 0; i < query_len; ++i) {
    query_on_shm[i] = query[i];  // not very efficient query loading temporally,
                                 // as assuming its not hotspot
  }
  __syncthreads();
  register int query_idx = 0;
  register int doc_len = doc_lens[tid];
  register int tmp_score = 0.;

  for (auto j = 0; j < doc_len; ++j) {
    while (query_idx < query_len &&
           query_on_shm[query_idx] < docs[tid * MAX_DOC_SIZE + j]) {
      ++query_idx;
    }
    if (query_idx < query_len) {
      tmp_score += (query_on_shm[query_idx] == docs[tid * MAX_DOC_SIZE + j]);
    }
  }
  scores[tid] = float(tmp_score) / max(query_len, doc_lens[tid]);  // tid
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
  // system_clock::time_point t0;
  // system_clock::time_point t1;
  // system_clock::time_point t2;
  // system_clock::time_point t3;
  // system_clock::time_point t4;
  // system_clock::time_point t5;
  // system_clock::time_point t6;

  // t0 = system_clock::now();
  // init
  // cudaSetDevice(0);
  // t0 = system_clock::now();
  size_t n_docs = docs.size();
  int query_num = querys.size();
  // printf("n_docs:%lu query_num:%d\n", n_docs, query_num);

  // std::vector<uint16_t *> doc_data_per_batch(BATCH_NUM, nullptr);
  // std::vector<int *> doc_len_per_batch(BATCH_NUM, nullptr);
  // std::vector<uint16_t *> d_doc_data_per_batch(BATCH_NUM, nullptr);
  // std::vector<int *> d_doc_len_per_batch(BATCH_NUM, nullptr);
  // std::vector<bool> h_batch_alloc(BATCH_NUM, false);
  // std::vector<bool> d_batch_alloc(BATCH_NUM, false);
  // std::vector<cudaStream_t> batch_streams(BATCH_NUM);
  // std::vector<std::thread> threads;
  std::vector<float *> query_d_scores(query_num, nullptr);
  std::vector<uint16_t *> query_data(query_num, nullptr);
  threads.reserve(NUM_THREADS);
  // t1 = system_clock::now();
  // for (int i = 0; i < BATCH_NUM; ++i) {
  //   cudaStreamCreate(&batch_streams[i]);
  // }
  for (int i = 0; i < query_num; ++i) {
    cudaMalloc(&query_d_scores[i], sizeof(float) * n_docs);
    cudaMalloc(&query_data[i], sizeof(uint16_t) * MAX_DOC_SIZE);
    cudaMemcpy(query_data[i], querys[i].data(),
               sizeof(uint16_t) * querys[i].size(), cudaMemcpyHostToDevice);
  }
  // t1 = system_clock::now();

  size_t done_docs_n = 0;
  int batch_id = 0;
  for (done_docs_n = 0; done_docs_n < n_docs;) {
    // printf("done_docs: %lu\n", done_docs_n);
    batch_id = batch_id % BATCH_NUM;
    cudaStreamSynchronize(batch_streams[batch_id]);
    // int start_docs = done_docs_n;
    // int end_docs =
    //     (start_docs + BATCH_SIZE > n_docs) ? n_docs : start_docs +
    //     BATCH_SIZE;
    size_t this_batch_n_docs = ((done_docs_n + BATCH_SIZE) >= n_docs)
                                   ? n_docs - done_docs_n
                                   : BATCH_SIZE;
    // printf("this_batch_n_docs: %lu\n", this_batch_n_docs);
    if (h_batch_alloc[batch_id] == false) {
      cudaMallocHost(&doc_data_per_batch[batch_id],
                     sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE);
      cudaMallocHost(&doc_len_per_batch[batch_id], sizeof(int) * BATCH_SIZE);
      h_batch_alloc[batch_id] = true;
    }
    // cudaStreamSynchronize(batch_streams[batch_id]);
    // if (d_doc_data_per_batch[batch_id] == nullptr) {
    //   cudaMallocAsync(&d_doc_data_per_batch[batch_id],
    //                   sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE,
    //                   batch_streams[batch_id]);
    //   cudaMallocAsync(&d_doc_len_per_batch[batch_id], sizeof(int) *
    //   BATCH_SIZE,
    //                   batch_streams[batch_id]);
    // }
    if (d_batch_alloc[batch_id] == false) {
      cudaMalloc(&d_doc_data_per_batch[batch_id],
                 sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE);
      cudaMalloc(&d_doc_len_per_batch[batch_id], sizeof(int) * BATCH_SIZE);
      d_batch_alloc[batch_id] = true;
    }
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
        for (int i = start_index; i < end_index; i++) {
          // for (int j = 0; j < docs[i + done_docs_n].size(); j++) {
          //   auto final_offset = i * MAX_DOC_SIZE + j;
          //   doc_data_per_batch[batch_id][final_offset] =
          //       docs[i + done_docs_n][j];
          // }
          memcpy(doc_data_per_batch[batch_id] + i * MAX_DOC_SIZE,
                 docs[i + done_docs_n].data(),
                 docs[i + done_docs_n].size() * sizeof(uint16_t));
          doc_len_per_batch[batch_id][i] = docs[i + done_docs_n].size();
        }
      });
    }
    for (auto &thread : threads) {
      thread.join();
    }
    // for (int i = 0; i < this_batch_n_docs; i++) {
    //   for (int j = 0; j < docs[i + done_docs_n].size(); j++) {
    //     auto group_sz = sizeof(group_t) / sizeof(uint16_t);
    //     auto layer_0_offset = j / group_sz;
    //     auto layer_0_stride = this_batch_n_docs * group_sz;
    //     auto layer_1_offset = i;
    //     auto layer_1_stride = group_sz;
    //     auto layer_2_offset = j % group_sz;
    //     auto final_offset = layer_0_offset * layer_0_stride +
    //                         layer_1_offset * layer_1_stride + layer_2_offset;
    //     doc_data_per_batch[batch_id][final_offset] = docs[i +
    //     done_docs_n][j];
    //   }
    //   doc_len_per_batch[batch_id][i] = docs[i + done_docs_n].size();
    // }

    // cudaMemsetAsync(d_doc_data_per_batch[batch_id], 0,
    //                 sizeof(uint16_t) * MAX_DOC_SIZE * BATCH_SIZE,
    //                 batch_streams[batch_id]);
    // cudaMemsetAsync(d_doc_len_per_batch[batch_id], 0, sizeof(int) *
    // BATCH_SIZE,
    //                 batch_streams[batch_id]);
    cudaMemcpyAsync(d_doc_data_per_batch[batch_id],
                    doc_data_per_batch[batch_id],
                    sizeof(uint16_t) * MAX_DOC_SIZE * this_batch_n_docs,
                    cudaMemcpyHostToDevice, batch_streams[batch_id]);
    cudaMemcpyAsync(d_doc_len_per_batch[batch_id], doc_len_per_batch[batch_id],
                    sizeof(int) * this_batch_n_docs, cudaMemcpyHostToDevice,
                    batch_streams[batch_id]);
    int block = N_THREADS_IN_ONE_BLOCK;
    int grid = (this_batch_n_docs + block - 1) / block;
    for (int query_id = 0; query_id < query_num; ++query_id) {
      docQueryScoringCoalescedMemoryAccessSampleKernel<<<
          grid, block, 0, batch_streams[batch_id]>>>(
          d_doc_data_per_batch[batch_id], d_doc_len_per_batch[batch_id],
          this_batch_n_docs, query_data[query_id], querys[query_id].size(),
          query_d_scores[query_id] + done_docs_n);
    }
    ++batch_id;
    done_docs_n += this_batch_n_docs;
  }
  // printf("done_docs: %lu\n", done_docs_n);
  // doc_data_per_batch.resize(BATCH_NUM);
  // t2 = system_clock::now();

  // 可以隐藏
  // std::vector<int> s_ans(TOPK);
  indices.reserve(query_num);
  // int *s_indices = nullptr;
  // cudaMallocHost(&s_indices, sizeof(int) * n_docs);
  // // t2 = system_clock::now();
  // std::iota(s_indices, s_indices + n_docs, 0);
  // t3 = system_clock::now();
  // std::vector<int> s2_indices(n_docs);
  // // // cudaSetDevice(0);
  // std::iota(s2_indices.begin(), s2_indices.end(), 0);
  // for (int i = 0; i < n_docs; ++i) {
  //   if (s_indices[i] != s2_indices[i]) {
  //     printf("seq wrong!\n");
  //     break;
  //   }
  // }
  std::vector<thrust::device_vector<int>> query_ranks(
      query_num, thrust::device_vector<int>(n_docs));
  // t3 = system_clock::now();
  cudaDeviceSynchronize();
  // for (int i = 0 ; i < BATCH_NUM; ++i) {
  //   cudaStreamSynchronize(batch_streams[batch_id]);
  // }
  for (int query_id = 0; query_id < query_num; ++query_id) {
    cudaMemcpy(thrust::raw_pointer_cast(query_ranks[query_id].data()),
               s_indices, n_docs * sizeof(int), cudaMemcpyHostToDevice);
    thrust::sort(query_ranks[query_id].begin(), query_ranks[query_id].end(),
                 cmp(query_d_scores[query_id]));
    for (size_t i = 0; i < TOPK; ++i) {
      s_ans[i] = query_ranks[query_id][i];
    }
    // cudaMemcpy(s_ans.data(),
    //            thrust::raw_pointer_cast(query_ranks[query_id].data()),
    //            sizeof(int) * TOPK, cudaMemcpyDeviceToHost);
    // std::cout<<"fine"<<std::endl;
    indices.push_back(s_ans);
  }
  // t4 = system_clock::now();
  // std::cout << "t1 - t0: " << duration_cast<milliseconds>(t1 - t0).count()
  //           << "ms" << std::endl;
  // std::cout << "t2 - t1: " << duration_cast<milliseconds>(t2 - t1).count()
  //           << "ms" << std::endl;
  // std::cout << "t3 - t2: " << duration_cast<milliseconds>(t3 - t2).count()
  //           << "ms" << std::endl;
  // std::cout << "t4 - t3: " << duration_cast<milliseconds>(t4 - t3).count()
  //           << "ms" << std::endl;
  // for (int query_id = 0; query_id < query_num; ++query_id) {
  //   cudaFree(query_d_scores[query_id]);
  //   cudaFree(query_data[query_id]);
  // }
  // for (int i = 0; i < BATCH_NUM; ++i) {
  //   cudaFreeHost(doc_data_per_batch[i]);
  //   cudaFreeHost(doc_len_per_batch[i]);
  //   cudaFree(d_doc_data_per_batch[i]);
  //   cudaFree(d_doc_len_per_batch[i]);
  //   cudaStreamDestroy(batch_streams[i]);
  // }
  // cudaFreeHost(s_indices);
}