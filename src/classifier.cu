#include <math.h>
#include <mpi.h>
#include <cassert>

#include "classifier.h"
#include "util.h"

#define PARAMETER_SIZE  (OFFSET21 + 4)
#define GPU_THREADS     8

#define CUDA_NEW(TENSOR_NAME, ...)                           \
  cudaMallocManaged((void **)&TENSOR_NAME, sizeof(Tensor));     \
  TENSOR_NAME->init_cuda({__VA_ARGS__});

#define CUDA_NEW_DATA(TENSOR_NAME, buf_, ...)               \
  cudaMallocManaged((void **)&TENSOR_NAME, sizeof(Tensor));     \
  TENSOR_NAME->init_cuda({__VA_ARGS__}, buf_);

#define CUDA_DELETE(TENSOR_NAME)                                \
  cudaFree(TENSOR_NAME->buf);                                   \
  cudaFree(TENSOR_NAME);

static int batch_size;
static int node_id, num_nodes;
static int num_devices;
static int article_size_per_node, articles_per_node;
static int article_size_per_device, articles_per_device;

// Multi-dimensional matrix containing fp32 elements
struct Tensor {
  Tensor(std::vector<int> shape_);
  Tensor(std::vector<int> shape_, float *buf_);
  ~Tensor();
  void init_cuda(std::vector<int> shape_);
  void init_cuda(std::vector<int> shape_, float *buf_);
  __host__ __device__ int num_elem();
  void fill_zeros();

  float *buf = nullptr;
  int ndim = 0;
  int shape[4];
};

Tensor::Tensor(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
}

Tensor::Tensor(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) { shape[i] = shape_[i]; }
  int N_ = num_elem();
  buf = (float *) calloc(N_, sizeof(float));
  for (int n = 0; n < N_; ++n) { buf[n] = buf_[n]; }
}

void Tensor::init_cuda(std::vector<int> shape_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) shape[i] = shape_[i];
  int N_ = num_elem();
  cudaMalloc((void **)&buf, sizeof(float) * N_);
}

void Tensor::init_cuda(std::vector<int> shape_, float *buf_) {
  ndim = shape_.size();
  for (int i = 0; i < ndim; ++i) shape[i] = shape_[i];
  int N_ = num_elem();
  cudaMalloc((void **)&buf, sizeof(float) * N_);
  cudaMemcpy(buf, buf_, sizeof(float) * N_, cudaMemcpyHostToDevice);
}

Tensor::~Tensor() {
  if (buf != nullptr) free(buf);
}

__host__ __device__ int Tensor::num_elem() {
  int sz = 1;
  for (int i = 0; i < ndim; ++i) { sz *= shape[i]; }
  return sz;
}

void Tensor::fill_zeros() {
  int N_ = num_elem();
  for (int n = 0; n < N_; ++n) { buf[n] = 0.0; }
}

Tensor **input_tensor;
float **out_buffer;

// Parameters
Tensor **w_conv1, **w_conv2, **w_conv3, **w_conv4, **w_conv5, **w_conv6, **b_conv1,
    **b_conv2, **b_conv3, **b_conv4, **b_conv5, **b_conv6, **w_fc1, **w_fc2, **w_fc3,
    **b_fc1, **b_fc2, **b_fc3, **gamma_conv1, **beta_conv1, **gamma_conv6, **beta_conv6;

// Activations
Tensor **a_conv1, **a_layernorm1, **a_relu1, **a_pool1;
Tensor **a_conv2, **a_relu2, **a_pool2;
Tensor **a_conv3, **a_relu3;
Tensor **a_conv4, **a_relu4;
Tensor **a_conv5, **a_relu5;
Tensor **a_conv6, **a_layernorm6, **a_relu6, **a_pool6;
Tensor **a_collapse;
Tensor **a_linear1, **a_relu7;
Tensor **a_linear2, **a_relu8;
Tensor **a_linear3;
Tensor **a_topone;

// Operations
void conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride, int padding, int dilation, bool has_bias);
void relu(Tensor *input, Tensor *output);
void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride);
void collapse(Tensor *input, Tensor *output);
void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias);
void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output);
void top_one(Tensor *input, Tensor *output);

void classifier(float *input_, float *output_, int N) {
  if (node_id != 0) {
    cudaMallocHost((void **)&input_, N * VOCAB_SIZE * MAX_LENGTH * sizeof(float));
    cudaMallocHost((void **)&output_, N * sizeof(float));
  }

  MPI_Scatter(
    input_ + node_id * article_size_per_node, article_size_per_node, MPI_FLOAT,
    input_ + node_id * article_size_per_node, article_size_per_node, MPI_FLOAT,
    0, MPI_COMM_WORLD);

  for (int dev_id = 0; dev_id < num_devices; dev_id++) {
    cudaSetDevice(dev_id);

    cudaMemcpy(
      input_tensor[dev_id]->buf,
      input_ + node_id * article_size_per_node + dev_id * article_size_per_device,
      sizeof(float) * batch_size * VOCAB_SIZE * MAX_LENGTH,
      cudaMemcpyHostToDevice);

    // Conv block 1 : Conv1d + LayerNorm + ReLU + MaxPool1d
    conv1d(input_tensor[dev_id], w_conv1[dev_id], b_conv1[dev_id], a_conv1[dev_id], 1, 0, 1, true);
    layernorm(a_conv1[dev_id], gamma_conv1[dev_id], beta_conv1[dev_id], a_layernorm1[dev_id]);
    relu(a_layernorm1[dev_id], a_relu1[dev_id]);
    maxpool1d(a_relu1[dev_id], a_pool1[dev_id], 3, 3);

    // Conv block 2 : Conv1d + ReLU + MaxPool1d
    conv1d(a_pool1[dev_id], w_conv2[dev_id], b_conv2[dev_id], a_conv2[dev_id], 1, 0, 1, true);
    relu(a_conv2[dev_id], a_relu2[dev_id]);
    maxpool1d(a_relu2[dev_id], a_pool2[dev_id], 3, 3);

    // Conv block 3 : Conv1d + ReLU
    conv1d(a_pool2[dev_id], w_conv3[dev_id], b_conv3[dev_id], a_conv3[dev_id], 1, 0, 1, true);
    relu(a_conv3[dev_id], a_relu3[dev_id]);

    // Conv block 4 : Conv1d + ReLU
    conv1d(a_relu3[dev_id], w_conv4[dev_id], b_conv4[dev_id], a_conv4[dev_id], 1, 0, 1, true);
    relu(a_conv4[dev_id], a_relu4[dev_id]);

    // Conv block 5 : Conv1d + ReLU
    conv1d(a_relu4[dev_id], w_conv5[dev_id], b_conv5[dev_id], a_conv5[dev_id], 1, 0, 1, true);
    relu(a_conv5[dev_id], a_relu5[dev_id]);

    // Conv block 6 : Conv1d + LayerNorm + ReLU + MaxPool1d
    conv1d(a_relu5[dev_id], w_conv6[dev_id], b_conv6[dev_id], a_conv6[dev_id], 1, 0, 1, true);
    layernorm(a_conv6[dev_id], gamma_conv6[dev_id], beta_conv6[dev_id], a_layernorm6[dev_id]);
    relu(a_layernorm6[dev_id], a_relu6[dev_id]);
    maxpool1d(a_relu6[dev_id], a_pool6[dev_id], 3, 3);

    // Collapse
    collapse(a_pool6[dev_id], a_collapse[dev_id]);

    // FC block 1 : Linear + ReLU
    linear(a_collapse[dev_id], w_fc1[dev_id], b_fc1[dev_id], a_linear1[dev_id], true);
    relu(a_linear1[dev_id], a_relu7[dev_id]);

    // FC block 2 : Linear + ReLU
    linear(a_relu7[dev_id], w_fc2[dev_id], b_fc2[dev_id], a_linear2[dev_id], true);
    relu(a_linear2[dev_id], a_relu8[dev_id]);

    // FC block 3 : Linear
    linear(a_relu8[dev_id], w_fc3[dev_id], b_fc3[dev_id], a_linear3[dev_id], true);

    top_one(a_linear3[dev_id], a_topone[dev_id]);

    cudaMemcpy(
      out_buffer[dev_id],
      a_topone[dev_id]->buf,
      batch_size * sizeof(float),
      cudaMemcpyDeviceToHost);

    for (int b = 0; b < batch_size; ++b) {
      output_[node_id * articles_per_node + dev_id * articles_per_device + b] = out_buffer[dev_id][b];
    }
  }
  
  MPI_Gather(
    output_ + node_id * articles_per_node, articles_per_node, MPI_FLOAT,
    output_ + node_id * articles_per_node, articles_per_node, MPI_FLOAT,
    0, MPI_COMM_WORLD);
}

__global__ void dev_conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride = 1, int padding = 0, int dilation = 1,
            bool has_bias = true) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int oc = blockIdx.y * blockDim.y + threadIdx.y;
  int ol = blockIdx.z * blockDim.z + threadIdx.z;

  int batch_size = input->shape[0];
  int out_channels = weight->shape[0];
  int in_channels = weight->shape[1];
  int kernel_size = weight->shape[2];
  int input_length = input->shape[2];
  int output_length =
      (input->shape[2] + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

  if (b >= batch_size || oc >= out_channels || ol >= output_length) return;

  int in_elem_per_batch = input->num_elem() / batch_size;
  int out_elem_per_batch = output->num_elem() / batch_size;

  float val = 0.0f;
  for (int ic = 0; ic < in_channels; ++ic) {
    for (int ks = 0; ks < kernel_size; ++ks) {
      float w = weight->buf[oc * in_channels * kernel_size + ic * kernel_size + ks];
      float i = input->buf[b * in_elem_per_batch + ic * input_length + ks + ol];
      val += w * i;
    }
  }
  if (has_bias) val += bias->buf[oc];
  output->buf[b * out_elem_per_batch + oc * output_length + ol] = val;
}

void conv1d(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            int stride = 1, int padding = 0, int dilation = 1,
            bool has_bias = true) {

  int out_channels = weight->shape[0];
  int output_length =
      (input->shape[2] + 2 * padding - dilation * (weight->shape[2] - 1) - 1) / stride + 1;

  dim3 blockDim(GPU_THREADS, GPU_THREADS, GPU_THREADS);
  dim3 gridDim(
    (batch_size + blockDim.x - 1) / blockDim.x,
    (out_channels + blockDim.y - 1) / blockDim.y,
    (output_length + blockDim.z - 1) / blockDim.z);
  dev_conv1d<<<gridDim, blockDim>>>(input, weight, bias, output, stride, padding, dilation, has_bias);
}

__global__ void dev_relu(Tensor *input, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int batch_size = input->shape[0];
  int elem_per_batch = input->num_elem() / batch_size;

  if (b >= batch_size || i >= elem_per_batch) return;

  int idx = b * elem_per_batch + i;
  float input_val = input->buf[idx];
  output->buf[idx] = input_val > 0.0f ? input_val : 0.0f;
}

void relu(Tensor *input, Tensor *output) {
  dim3 blockDim(GPU_THREADS, GPU_THREADS);
  dim3 gridDim((batch_size + blockDim.x - 1) / blockDim.x, (input->num_elem() / batch_size + blockDim.y - 1) / blockDim.y);
  dev_relu<<<gridDim, blockDim>>>(input, output);
}

__global__ void dev_maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int oc = blockIdx.y * blockDim.y + threadIdx.y;
  int ol = blockIdx.z * blockDim.z + threadIdx.z;

  int batch_size = input->shape[0];
  int IC = input->shape[1];
  int IL = input->shape[2];
  int OC = output->shape[1];
  int OL = output->shape[2];

  if (b >= batch_size || oc >= OC || ol >= OL) return;

  float mx = -1e99;
  for (int ks = 0; ks < kernel_size; ++ks) {
    float val = input->buf[(b * IC + oc) * IL + ks + ol * stride];
    if (val > mx) mx = val;
  }

  output->buf[(b * OC + oc) * OL + ol] = mx;
}

void maxpool1d(Tensor *input, Tensor *output, int kernel_size, int stride) {
  dim3 blockDim(GPU_THREADS, GPU_THREADS, GPU_THREADS);
  dim3 gridDim(
    (batch_size + blockDim.x - 1) / blockDim.x,
    (output->shape[1] + blockDim.y - 1) / blockDim.y,
    (output->shape[2] + blockDim.z - 1) / blockDim.z);
  dev_maxpool1d<<<gridDim, blockDim>>>(input, output, kernel_size, stride);
}

__global__ void dev_collapse(Tensor *input, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y * blockDim.y + threadIdx.y;

  int batch_size = input->shape[0];
  int elem_per_batch = input->num_elem() / batch_size;

  if (b >= batch_size || i >= elem_per_batch) return;

  int idx = b * elem_per_batch + i;
  output->buf[idx] = input->buf[idx];
}

void collapse(Tensor *input, Tensor *output) {
  int elem_per_batch = input->num_elem() / batch_size;

  dim3 blockDim(GPU_THREADS, GPU_THREADS);
  dim3 gridDim(
    (batch_size + blockDim.x - 1) / blockDim.x,
    (elem_per_batch + blockDim.y - 1) / blockDim.y);
  dev_collapse<<<gridDim, blockDim>>>(input, output);
}

__global__ void dev_linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;
  int oc = blockIdx.y * blockDim.y + threadIdx.y;

  int batch_size = input->shape[0];
  int IC = input->shape[1];
  int OC = output->shape[1];

  if (b >= batch_size || oc >= OC) return;

  float val = 0.0;
  for (int ic = 0; ic < IC; ++ic) {
    val += input->buf[b * IC + ic] * weight->buf[oc * IC + ic];
  }
  if (has_bias) val += bias->buf[oc];
  output->buf[b * OC + oc] = val;
}

void linear(Tensor *input, Tensor *weight, Tensor *bias, Tensor *output,
            bool has_bias) {
  dim3 blockDim(GPU_THREADS, GPU_THREADS);
  dim3 gridDim(
    (batch_size + blockDim.x - 1) / blockDim.x,
    (output->shape[1] + blockDim.y - 1) / blockDim.y);
  dev_linear<<<gridDim, blockDim>>>(input, weight, bias, output, has_bias);
}

__global__ void dev_layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output) {
  // E[X], E[X^2]
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_size = input->shape[0];
  int elem_per_batch = input->num_elem() / batch_size;

  if (b >= batch_size) return;

  float sum1 = 0.0f, sum2 = 0.0f;
  for (int i = 0; i < elem_per_batch; ++i) {
    int idx = b * elem_per_batch + i;
    float val = input->buf[idx];
    float val_square = val * val;
    sum1 += val;
    sum2 += val_square;
  }
  float mean1 = sum1 / (float)elem_per_batch;
  float mean2 = sum2 / (float)elem_per_batch;

  // V[X]
  float var = mean2 - mean1 * mean1; 

  // Normalization
  for (int i = 0; i < elem_per_batch; ++i) {
    int idx = b * elem_per_batch + i;
    float in = input->buf[idx];
    float ga = gamma->buf[i];
    float be = beta->buf[i];
    output->buf[idx] = (in - mean1) / sqrtf(var + 1e-5) * ga + be;
  }
}

void layernorm(Tensor *input, Tensor *gamma, Tensor *beta, Tensor *output) {
  dim3 blockDim(GPU_THREADS);
  dim3 gridDim(
    (batch_size + blockDim.x - 1) / blockDim.x);
  dev_layernorm<<<gridDim, blockDim>>>(input, gamma, beta, output);
}

__global__ void dev_top_one(Tensor *input, Tensor *output) {
  int b = blockIdx.x * blockDim.x + threadIdx.x;

  int batch_size = input->shape[0];
  int IC = input->shape[1];
  int elem_per_batch = input->num_elem() / batch_size;

  if (b >= batch_size) return;

  float max_val = -1e99f;
  int max_idx = 0;
  for (int i = 0; i < IC; ++i) {
    float val = input->buf[b * IC + i];
    if (val > max_val) {
      max_val = val;
      max_idx = i;
    }
  }

  output->buf[b] = max_idx;
}

void top_one(Tensor *input, Tensor *output) {
  dim3 blockDim(GPU_THREADS);
  dim3 gridDim(
    (batch_size + blockDim.x - 1) / blockDim.x);
  dev_top_one<<<gridDim, blockDim>>>(input, output);
}

void initialize_classifier(float *parameter, int N) {
  MPI_Comm_rank(MPI_COMM_WORLD, &node_id);
  MPI_Comm_size(MPI_COMM_WORLD, &num_nodes);

  cudaGetDeviceCount(&num_devices);

  articles_per_node = N / num_nodes;
  article_size_per_node = articles_per_node * VOCAB_SIZE * MAX_LENGTH;
  articles_per_device = articles_per_node / num_devices;
  article_size_per_device = articles_per_device * VOCAB_SIZE * MAX_LENGTH;

  batch_size = articles_per_device;

  if (node_id != 0) {
    cudaMallocHost((void **)&parameter, PARAMETER_SIZE * sizeof(float));
  }
  MPI_Bcast(parameter, PARAMETER_SIZE, MPI_FLOAT, 0, MPI_COMM_WORLD);

  w_conv1 = new Tensor *[num_devices];
  b_conv1 = new Tensor *[num_devices];
  gamma_conv1 = new Tensor *[num_devices];
  beta_conv1 = new Tensor *[num_devices];
  w_conv2 = new Tensor *[num_devices];
  b_conv2 = new Tensor *[num_devices];
  w_conv3 = new Tensor *[num_devices];
  b_conv3 = new Tensor *[num_devices];
  w_conv4 = new Tensor *[num_devices];
  b_conv4 = new Tensor *[num_devices];
  w_conv5 = new Tensor *[num_devices];
  b_conv5 = new Tensor *[num_devices];
  w_conv6 = new Tensor *[num_devices];
  b_conv6 = new Tensor *[num_devices];
  gamma_conv6 = new Tensor *[num_devices];
  beta_conv6 = new Tensor *[num_devices];
  w_fc1 = new Tensor *[num_devices];
  b_fc1 = new Tensor *[num_devices];
  w_fc2 = new Tensor *[num_devices];
  b_fc2 = new Tensor *[num_devices];
  w_fc3 = new Tensor *[num_devices];
  b_fc3 = new Tensor *[num_devices];

  input_tensor = new Tensor *[num_devices];

  a_conv1 = new Tensor *[num_devices];
  a_layernorm1 = new Tensor *[num_devices];
  a_relu1 = new Tensor *[num_devices];
  a_pool1 = new Tensor *[num_devices];
  a_conv2 = new Tensor *[num_devices];
  a_relu2 = new Tensor *[num_devices];
  a_pool2 = new Tensor *[num_devices];
  a_conv3 = new Tensor *[num_devices];
  a_relu3 = new Tensor *[num_devices];
  a_conv4 = new Tensor *[num_devices];
  a_relu4 = new Tensor *[num_devices];
  a_conv5 = new Tensor *[num_devices];
  a_relu5 = new Tensor *[num_devices];
  a_conv6 = new Tensor *[num_devices];
  a_layernorm6 = new Tensor *[num_devices];
  a_relu6 = new Tensor *[num_devices];
  a_pool6 = new Tensor *[num_devices];
  a_collapse = new Tensor *[num_devices];
  a_linear1 = new Tensor *[num_devices];
  a_relu7 = new Tensor *[num_devices];
  a_linear2 = new Tensor *[num_devices];
  a_relu8 = new Tensor *[num_devices];
  a_linear3 = new Tensor *[num_devices];
  a_topone = new Tensor *[num_devices];

  out_buffer = new float *[num_devices];

  for (int dev_id = 0; dev_id < num_devices; dev_id++) {
    cudaSetDevice(dev_id);

    CUDA_NEW_DATA(w_conv1[dev_id], parameter + OFFSET0, 256, 70, 7);
    CUDA_NEW_DATA(b_conv1[dev_id], parameter + OFFSET1, 256);
    CUDA_NEW_DATA(gamma_conv1[dev_id], parameter + OFFSET2, 256, 1008);
    CUDA_NEW_DATA(beta_conv1[dev_id], parameter + OFFSET3, 256, 1008);
    CUDA_NEW_DATA(w_conv2[dev_id], parameter + OFFSET4, 256, 256, 7);
    CUDA_NEW_DATA(b_conv2[dev_id], parameter + OFFSET5, 256);
    CUDA_NEW_DATA(w_conv3[dev_id], parameter + OFFSET6, 256, 256, 3);
    CUDA_NEW_DATA(b_conv3[dev_id], parameter + OFFSET7, 256);
    CUDA_NEW_DATA(w_conv4[dev_id], parameter + OFFSET8, 256, 256, 3);
    CUDA_NEW_DATA(b_conv4[dev_id], parameter + OFFSET9, 256);
    CUDA_NEW_DATA(w_conv5[dev_id], parameter + OFFSET10, 256, 256, 3);
    CUDA_NEW_DATA(b_conv5[dev_id], parameter + OFFSET11, 256);
    CUDA_NEW_DATA(w_conv6[dev_id], parameter + OFFSET12, 256, 256, 3);
    CUDA_NEW_DATA(b_conv6[dev_id], parameter + OFFSET13, 256);
    CUDA_NEW_DATA(gamma_conv6[dev_id], parameter + OFFSET14, 256, 102);
    CUDA_NEW_DATA(beta_conv6[dev_id], parameter + OFFSET15, 256, 102);
    CUDA_NEW_DATA(w_fc1[dev_id], parameter + OFFSET16, 1024, 8704);
    CUDA_NEW_DATA(b_fc1[dev_id], parameter + OFFSET17, 1024);
    CUDA_NEW_DATA(w_fc2[dev_id], parameter + OFFSET18, 1024, 1024);
    CUDA_NEW_DATA(b_fc2[dev_id], parameter + OFFSET19, 1024);
    CUDA_NEW_DATA(w_fc3[dev_id], parameter + OFFSET20, 4, 1024);
    CUDA_NEW_DATA(b_fc3[dev_id], parameter + OFFSET21, 4);

    CUDA_NEW(input_tensor[dev_id], batch_size, VOCAB_SIZE, MAX_LENGTH);

    CUDA_NEW(a_conv1[dev_id], batch_size, 256, 1008);
    CUDA_NEW(a_layernorm1[dev_id], batch_size, 256, 1008);
    CUDA_NEW(a_relu1[dev_id], batch_size, 256, 1008);
    CUDA_NEW(a_pool1[dev_id], batch_size, 256, 336);
    CUDA_NEW(a_conv2[dev_id], batch_size, 256, 330);
    CUDA_NEW(a_relu2[dev_id], batch_size, 256, 330);
    CUDA_NEW(a_pool2[dev_id], batch_size, 256, 110);
    CUDA_NEW(a_conv3[dev_id], batch_size, 256, 108);
    CUDA_NEW(a_relu3[dev_id], batch_size, 256, 108);
    CUDA_NEW(a_conv4[dev_id], batch_size, 256, 106);
    CUDA_NEW(a_relu4[dev_id], batch_size, 256, 106);
    CUDA_NEW(a_conv5[dev_id], batch_size, 256, 104);
    CUDA_NEW(a_relu5[dev_id], batch_size, 256, 104);
    CUDA_NEW(a_conv6[dev_id], batch_size, 256, 102);
    CUDA_NEW(a_layernorm6[dev_id], batch_size, 256, 102);
    CUDA_NEW(a_relu6[dev_id], batch_size, 256, 102);
    CUDA_NEW(a_pool6[dev_id], batch_size, 256, 34);
    CUDA_NEW(a_collapse[dev_id], batch_size, 8704);
    CUDA_NEW(a_linear1[dev_id], batch_size, 1024);
    CUDA_NEW(a_relu7[dev_id], batch_size, 1024);
    CUDA_NEW(a_linear2[dev_id], batch_size, 1024);
    CUDA_NEW(a_relu8[dev_id], batch_size, 1024);
    CUDA_NEW(a_linear3[dev_id], batch_size, 4);
    CUDA_NEW(a_topone[dev_id], batch_size);

    out_buffer[dev_id] = new float[batch_size];
  }
}

// Free all dynamically allocated variables
void finalize_classifier() {
  for (int dev_id = 0; dev_id < num_devices; dev_id++) {
    cudaSetDevice(dev_id);
    CUDA_DELETE(w_conv1[dev_id]);
    CUDA_DELETE(b_conv1[dev_id]);
    CUDA_DELETE(w_conv2[dev_id]);
    CUDA_DELETE(b_conv2[dev_id]);
    CUDA_DELETE(w_conv3[dev_id]);
    CUDA_DELETE(b_conv3[dev_id]);
    CUDA_DELETE(w_conv4[dev_id]);
    CUDA_DELETE(b_conv4[dev_id]);
    CUDA_DELETE(w_conv5[dev_id]);
    CUDA_DELETE(b_conv5[dev_id]);
    CUDA_DELETE(w_conv6[dev_id]);
    CUDA_DELETE(b_conv6[dev_id]);
    CUDA_DELETE(w_fc1[dev_id]);
    CUDA_DELETE(b_fc1[dev_id]);
    CUDA_DELETE(w_fc2[dev_id]);
    CUDA_DELETE(b_fc2[dev_id]);
    CUDA_DELETE(w_fc3[dev_id]);
    CUDA_DELETE(b_fc3[dev_id]);
    CUDA_DELETE(gamma_conv1[dev_id]);
    CUDA_DELETE(gamma_conv6[dev_id]);
    CUDA_DELETE(beta_conv1[dev_id]);
    CUDA_DELETE(beta_conv6[dev_id]);

    CUDA_DELETE(input_tensor[dev_id]);

    CUDA_DELETE(a_conv1[dev_id]);
    CUDA_DELETE(a_layernorm1[dev_id]);
    CUDA_DELETE(a_relu1[dev_id]);
    CUDA_DELETE(a_pool1[dev_id]);
    CUDA_DELETE(a_conv2[dev_id]);
    CUDA_DELETE(a_relu2[dev_id]);
    CUDA_DELETE(a_pool2[dev_id]);
    CUDA_DELETE(a_conv3[dev_id]);
    CUDA_DELETE(a_relu3[dev_id]);
    CUDA_DELETE(a_conv4[dev_id]);
    CUDA_DELETE(a_relu4[dev_id]);
    CUDA_DELETE(a_conv5[dev_id]);
    CUDA_DELETE(a_relu5[dev_id]);
    CUDA_DELETE(a_conv6[dev_id]);
    CUDA_DELETE(a_layernorm6[dev_id]);
    CUDA_DELETE(a_relu6[dev_id]);
    CUDA_DELETE(a_pool6[dev_id]);
    CUDA_DELETE(a_collapse[dev_id]);
    CUDA_DELETE(a_linear1[dev_id]);
    CUDA_DELETE(a_relu7[dev_id]);
    CUDA_DELETE(a_linear2[dev_id]);
    CUDA_DELETE(a_relu8[dev_id]);
    CUDA_DELETE(a_linear3[dev_id]);
    CUDA_DELETE(a_topone[dev_id]);

    delete[] out_buffer[dev_id];
  }

  delete[] w_conv1;
  delete[] b_conv1;
  delete[] w_conv2;
  delete[] b_conv2;
  delete[] w_conv3;
  delete[] b_conv3;
  delete[] w_conv4;
  delete[] b_conv4;
  delete[] w_conv5;
  delete[] b_conv5;
  delete[] w_conv6;
  delete[] b_conv6;
  delete[] w_fc1;
  delete[] b_fc1;
  delete[] w_fc2;
  delete[] b_fc2;
  delete[] w_fc3;
  delete[] b_fc3;
  delete[] gamma_conv1;
  delete[] gamma_conv6;
  delete[] beta_conv1;
  delete[] beta_conv6;

  delete[] input_tensor;

  delete[] a_conv1;
  delete[] a_layernorm1;
  delete[] a_relu1;
  delete[] a_pool1;
  delete[] a_conv2;
  delete[] a_relu2;
  delete[] a_pool2;
  delete[] a_conv3;
  delete[] a_relu3;
  delete[] a_conv4;
  delete[] a_relu4;
  delete[] a_conv5;
  delete[] a_relu5;
  delete[] a_conv6;
  delete[] a_layernorm6;
  delete[] a_relu6;
  delete[] a_pool6;
  delete[] a_collapse;
  delete[] a_linear1;
  delete[] a_relu7;
  delete[] a_linear2;
  delete[] a_relu8;
  delete[] a_linear3;
  delete[] a_topone;

  delete[] out_buffer;
}
