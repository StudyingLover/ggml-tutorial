#include "ggml.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>

#include <algorithm>
#include <string>
#include <vector>

struct mnist_cnn {
  struct ggml_tensor *conv1_weight;
  struct ggml_tensor *conv1_bias;
  struct ggml_tensor *conv2_weight;
  struct ggml_tensor *conv2_bias;
  struct ggml_tensor *fc1_weight;
  struct ggml_tensor *fc1_bias;
  struct ggml_tensor *fc2_weight;
  struct ggml_tensor *fc2_bias;
  struct ggml_context *ctx;
};

bool mnist_cnn_load(const std::string &fname, mnist_cnn &model) {
  struct gguf_init_params params = {
      /*.no_alloc   =*/false,
      /*.ctx        =*/&model.ctx,
  };
  gguf_context *ctx = gguf_init_from_file(fname.c_str(), params);
  if (!ctx) {
    fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
    return false;
  }
  model.conv1_weight = ggml_get_tensor(model.ctx, "conv1_weight");
  model.conv1_bias = ggml_get_tensor(model.ctx, "conv1_bias");
  model.conv2_weight = ggml_get_tensor(model.ctx, "conv2_weight");
  model.conv2_bias = ggml_get_tensor(model.ctx, "conv2_bias");
  model.fc1_weight = ggml_get_tensor(model.ctx, "fc1_weight");
  model.fc1_bias = ggml_get_tensor(model.ctx, "fc1_bias");
  model.fc2_weight = ggml_get_tensor(model.ctx, "fc2_weight");
  model.fc2_bias = ggml_get_tensor(model.ctx, "fc2_bias");
  return true;
}

void print_tensor_shape(const ggml_tensor *t) {
  printf("tensor named '%s' has shape: %lld %lld %lld %lld\n", t->name,
         (long long)t->ne[0], (long long)t->ne[1], (long long)t->ne[2],
         (long long)t->ne[3]);
}
int mnist_eval(const mnist_cnn &model, const int n_threads,
               std::vector<float> digit, const char *fname_cgraph) {
  static size_t buf_size = 100000 * sizeof(float) * 4;
  static void *buf = malloc(buf_size);

  struct ggml_init_params params = {
      /*.mem_size   =*/buf_size,
      /*.mem_buffer =*/buf,
      /*.no_alloc   =*/false,
  };

  struct ggml_context *ctx0 = ggml_init(params);
  struct ggml_cgraph *gf = ggml_new_graph(ctx0);

  struct ggml_tensor *input =
      ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 28, 28, 1, 1);

  ggml_tensor *conv1_weight =
      ggml_reshape_4d(ctx0, model.conv1_weight, 3, 3, 1, 32);
  ggml_tensor *conv1_bias =
      ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 28, 28, 32, 1);

  // 现在conv1_bias形状是32 1 1 1，需要将他broadcast成28 28 32 1
  for (int i = 0; i < 28; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 32; k++) {
        ((float *)conv1_bias->data)[i * 28 * 32 + j * 32 + k] =
            ggml_get_data_f32(model.conv1_bias)[k];
      }
    }
    conv1_bias = ggml_reshape_4d(ctx0, conv1_bias, 28, 28, 32, 1);
  }

  ggml_tensor *conv2_bias =
      ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 14, 14, 64, 1);

  // 现在conv2_bias形状是64 1 1 1，需要将他broadcast成14 14 64 1
  for (int i = 0; i < 14; i++) {
    for (int j = 0; j < 14; j++) {
      for (int k = 0; k < 64; k++) {
        ((float *)conv2_bias->data)[i * 14 * 64 + j * 64 + k] =
            ggml_get_data_f32(model.conv2_bias)[k];
      }
    }
    conv2_bias = ggml_reshape_4d(ctx0, conv2_bias, 14, 14, 64, 1);
  }

  memcpy(input->data, digit.data(), ggml_nbytes(input));
  ggml_set_name(input, "input");

  fprintf(stderr, "%s: building forward\n", __func__);

  ggml_tensor *cur =
      ggml_conv_2d(ctx0, conv1_weight, input, 1, 1, 1, 1, 1,
                   1); // [3, 3, 1, 32] [28, 28, 1, 1] -> [28 28 32 1]

  cur = ggml_add(ctx0, cur, conv1_bias);
  fprintf(stderr, "2\n");
  cur = ggml_relu(ctx0, cur);
  fprintf(stderr, "3\n");
  cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  fprintf(stderr, "4\n");
  print_tensor_shape(model.conv2_weight);
  print_tensor_shape(cur);
  cur = ggml_conv_2d(ctx0, model.conv2_weight, cur, 1, 1, 1, 1, 1, 1);
  fprintf(stderr, "5\n");
  print_tensor_shape(cur);
  print_tensor_shape(conv2_bias);
  cur = ggml_add(ctx0, cur, conv2_bias);
  fprintf(stderr, "6\n");
  cur = ggml_relu(ctx0, cur);
  fprintf(stderr, "7\n");
  cur = ggml_pool_2d(ctx0, cur, GGML_OP_POOL_MAX, 2, 2, 2, 2, 0, 0);
  fprintf(stderr, "8\n");
  cur = ggml_reshape_2d(ctx0, cur, 7 * 7 * 64, 1);
  fprintf(stderr, "9\n");
  cur = ggml_mul_mat(ctx0, model.fc1_weight, cur);
  fprintf(stderr, "10\n");
  cur = ggml_add(ctx0, cur, model.fc1_bias);
  fprintf(stderr, "11\n");
  cur = ggml_relu(ctx0, cur);
  fprintf(stderr, "12\n");
  cur = ggml_mul_mat(ctx0, model.fc2_weight, cur);
  fprintf(stderr, "13\n");
  cur = ggml_add(ctx0, cur, model.fc2_bias);
  fprintf(stderr, "14\n");

  ggml_tensor *result = cur;
  ggml_set_name(result, "result");

  // fprintf(stderr, "%s: building forward expand\n", __func__);

  ggml_build_forward_expand(gf, result);
  ggml_graph_compute_with_ctx(ctx0, gf, n_threads);

  // ggml_graph_dump_dot(gf, NULL, "mnist-cnn.dot");

  if (fname_cgraph) {
    ggml_graph_export(gf, fname_cgraph);

    // fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__,
    //         fname_cgraph);
  }

  const float *probs_data = ggml_get_data_f32(result);
  const int prediction =
      std::max_element(probs_data, probs_data + 10) - probs_data;
  ggml_free(ctx0);
  return prediction;
}

struct image_u8 {
  int nx;
  int ny;

  std::vector<uint8_t> data;
};

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION

bool image_load_from_file(const std::string &fname, image_u8 &img) {
  int nx, ny, nc;
  auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
  if (!data) {
    fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
    return false;
  }

  img.nx = nx;
  img.ny = ny;
  img.data.resize(nx * ny * 1);
  memcpy(img.data.data(), data, nx * ny * 1);

  stbi_image_free(data);

  return true;
}

int main(int argc, char **argv) {
  fprintf(stderr, "main\n");
  srand(time(NULL));
  ggml_time_init();

  if (argc != 3) {
    fprintf(stderr,
            "Usage: %s <model.gguf> <image.jpg>\n"
            "  <model.gguf>  - path to the model file\n"
            "  <image.jpg>   - path to the image file\n",
            argv[0]);
    return 1;
  }

  const std::string model_fname = argv[1];
  const std::string image_fname = argv[2];

  mnist_cnn model;
  if (!mnist_cnn_load(model_fname, model)) {
    fprintf(stderr, "%s: failed to load the model\n", argv[0]);
    return 1;
  } else {
    fprintf(stderr, "%s: loaded the model\n", argv[0]);
  }

  image_u8 img;
  if (!image_load_from_file(image_fname, img)) {
    fprintf(stderr, "%s: failed to load the image\n", argv[0]);
    return 1;
  } else {
    fprintf(stderr, "%s: loaded the image\n", argv[0]);
  }

  std::vector<float> digit(img.nx * img.ny);
  for (int i = 0; i < img.nx * img.ny; i++) {
    digit[i] = img.data[i] / 255.0f;
  }

  fprintf(stderr, "%s: loaded the digit\n", argv[0]);

  const int prediction = mnist_eval(model, 1, digit, "mnist-cnn-cgraph.dot");
  printf("Prediction: %d\n", prediction);

  return 0;
}