# mnist-torch
MNIST手写体识别是经典的机器学习问题，可以被称作机器学习的hello world了，我希望通过mnist来作为系列教程的第一节，来介绍如何使用ggml量化，推理一个模型。这个教程将会使用pytorch来训练一个简单的全连接神经网络，然后使用ggml量化，最后使用ggml推理这个模型。

## 训练模型 
首先我们使用pytorch来训练一个简单的全连接神经网络，代码在`train.py` 文件中，训练好的模型会被保存到`model/mnist_model.pth` 文件中。代码是非常简单的torch代码

这里我们需要强调一下模型结构
```python
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```
模型由两个全连接层组成，第一个全连接层的输入是784维，输出是128维，第二个全连接层的输入是128维，输出是10维。我们需要知道这个结构，因为我们需要在量化模型时知道各个层的名字。

前向传播过程是先将输入reshape成2d的张量，然后进行矩阵乘法，然后加上偏置，然后relu，然后再进行矩阵乘法，然后再加上偏置，最后得到结果。

## 量化
我们需要使用ggml对模型进行量化，代码在`convert-pth-to-ggml.py` 文件中,使用`python convert-pth-to-ggml.py model/mnist_model.pth`进行转换，量化后的模型会被保存到`model/mnist-ggml-model-f32.pth` 文件中。

这里需要对很多细节作出解释：
1. ggml量化的模型格式叫做gguf,文件开头有一个魔数标记了这个文件是gguf文件，接下来是模型的各种数据，具体细节可以查看[官方文档](https://github.com/ggerganov/ggml/blob/master/docs/gguf.md)。为了方便，作者提供了一个python库来读写gguf文件，使用`pip install gguf` 就可以安装。
2. 我们需要知道模型中各个层数据的名字，使用`model.keys()` 就可以知道了。知道各个层的名字之后我们就可以取出各个层的数据，并对需要的层进行量化，也就是下面这段代码，我对weights进行了量化，转换成了`float16`
```
fc1_weights = model["fc1.weight"].data.numpy()
fc1_weights = fc1_weights.astype(np.float16)
gguf_writer.add_tensor("fc1_weights", fc1_weights, raw_shape=(128, 784))

fc1_bias = model["fc1.bias"].data.numpy()
gguf_writer.add_tensor("fc1_bias", fc1_bias)

fc2_weights = model["fc2.weight"].data.numpy()
fc2_weights = fc2_weights.astype(np.float16)
gguf_writer.add_tensor("fc2_weights", fc2_weights, raw_shape=(10, 128))

fc2_bias = model["fc2.bias"].data.numpy()
gguf_writer.add_tensor("fc2_bias", fc2_bias)
```

3. 保存模型按照代码特定顺序执行就可以了
```
gguf_writer = gguf.GGUFWriter(fname_out, "simple-nn")

fc1_weights = model["fc1.weight"].data.numpy()
fc1_weights = fc1_weights.astype(np.float16)
gguf_writer.add_tensor("fc1_weights", fc1_weights, raw_shape=(128, 784))

fc1_bias = model["fc1.bias"].data.numpy()
gguf_writer.add_tensor("fc1_bias", fc1_bias)

fc2_weights = model["fc2.weight"].data.numpy()
fc2_weights = fc2_weights.astype(np.float16)
gguf_writer.add_tensor("fc2_weights", fc2_weights, raw_shape=(10, 128))

fc2_bias = model["fc2.bias"].data.numpy()
gguf_writer.add_tensor("fc2_bias", fc2_bias)

gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()
```

我们可以看到，原本模型大小是399.18kb,现在的大小是199.31kb，确实是缩小了很多的。

## 推理
使用ggml推理实际上是对代码能力和机器学习理论功底的一个综合考察，因为你不仅需要能写c++代码，还要会用ggml提供的各种张量操作实现模型的前向传播进行推理，如果你不了解模型是怎么进行计算的，这里很容易不会写。我们接下来详细来说怎么写代码。

首先按照我们torch定义的模型，我们定义一个结构体来存储模型权重
```c++
struct mnist_model {
    struct ggml_tensor * fc1_weight;
    struct ggml_tensor * fc1_bias;
    struct ggml_tensor * fc2_weight;
    struct ggml_tensor * fc2_bias;
    struct ggml_context * ctx;
};
```

接下来加载模型,传入两个参数，模型地址和模型结构体。gguf_init_params 是模型初始化时的两个参数，分别代表是否**不加载模型**(实际含义是如果提供的gguf_context是no_alloc，则我们创建“空”张量并不读取二进制文件。否则，我们还将二进制文件加载到创建的ggml_context中，并将ggml_tensor结构体的"data"成员指向二进制文件中的适当位置。)和模型的地址。gguf_init_from_file 函数会返回一个gguf_context，这个结构体包含了模型的所有信息，我们需要从中取出我们需要的张量，这里我们需要的张量是fc1_weight,fc1_bias,fc2_weight,fc2_bias(和量化模型时保持一致)。
```c++
bool mnist_model_load(const std::string & fname, mnist_model & model) {
    struct gguf_init_params params = {
        /*.no_alloc   =*/ false,
        /*.ctx        =*/ &model.ctx,
    };
    gguf_context * ctx = gguf_init_from_file(fname.c_str(), params);
    if (!ctx) {
        fprintf(stderr, "%s: gguf_init_from_file() failed\n", __func__);
        return false;
    }
    model.fc1_weight = ggml_get_tensor(model.ctx, "fc1_weights");
    model.fc1_bias = ggml_get_tensor(model.ctx, "fc1_bias");
    model.fc2_weight = ggml_get_tensor(model.ctx, "fc2_weights");
    model.fc2_bias = ggml_get_tensor(model.ctx, "fc2_bias");
    return true;
}
```

接下来我们写模型的前向传播,完整代码在`main-torch.cpp`。传入的参数是模型的地址，线程数，数据和是否导出计算图(这个我们先不讨论)。

首先初始化模型和数据
```c++
static size_t buf_size = 100000 * sizeof(float) * 4;
static void * buf = malloc(buf_size);

struct ggml_init_params params = {
    /*.mem_size   =*/ buf_size,
    /*.mem_buffer =*/ buf,
    /*.no_alloc   =*/ false,
};

struct ggml_context * ctx0 = ggml_init(params);
struct ggml_cgraph * gf = ggml_new_graph(ctx0);
```

我们先复习一下全连接层的计算。每个全连接层有两个参数$W$和$B$，对于一个输出数据$X$,只需要$WX+B$就是一层前向传播的结果。

那么我们先初始化一个4d的张量作为输入(和torch很像)，然后将数据复制到这个张量中，然后将这个张量reshape成2d的张量，然后进行矩阵乘法，然后加上偏置，然后relu，然后再进行矩阵乘法，然后再加上偏置，最后得到结果。

```c++
struct ggml_tensor * input = ggml_new_tensor_4d(ctx0, GGML_TYPE_F32, 28, 28, 1, 1);
    memcpy(input->data, digit.data(), ggml_nbytes(input));
    ggml_set_name(input, "input");
    ggml_tensor * cur = ggml_reshape_2d(ctx0, input, 784, 1);
    // std::cout<<model.fc1_weight->data;
    cur = ggml_mul_mat(ctx0, model.fc1_weight, cur);
    // printf("%d",ggml_can_mul_mat(model.fc1_weight, cur));
    // cur = ggml_mul_mat(ctx0, cur, model.fc1_weight);
    cur = ggml_add(ctx0, cur, model.fc1_bias);
    cur = ggml_relu(ctx0, cur);
    cur = ggml_mul_mat(ctx0, model.fc2_weight, cur);
    cur = ggml_add(ctx0, cur, model.fc2_bias);
```

接下来通过计算图计算出结果，ggml已经提供了api
```
ggml_build_forward_expand(gf, result);
ggml_graph_compute_with_ctx(ctx0, gf, n_threads);
```


我们需要将结果reshape成1d的张量，然后取出最大值，这个最大值就是我们的预测结果。
```c++
const int prediction = std::max_element(probs_data, probs_data + 10) - probs_data;
const float * probs_data = ggml_get_data_f32(result);
```

我们可以将计算图进行存储,这部分代码我们先不讨论
```c++
//ggml_graph_print(&gf);
ggml_graph_dump_dot(gf, NULL, "mnist-cnn.dot");

if (fname_cgraph) {
    // export the compute graph for later use
    // see the "mnist-cpu" example
    ggml_graph_export(gf, fname_cgraph);

    fprintf(stderr, "%s: exported compute graph to '%s'\n", __func__, fname_cgraph);
}
```

最后记得释放内存
```
ggml_free(ctx0);
```

## 图片读取
我们这里要用到`stb_image.h`这个头文件，我们通过下面的代码导入
```c++
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
```

我们定义一个结构体来存储图片
```c++
struct image_u8 {
    int nx;
    int ny;

    std::vector<uint8_t> data;
};
```

接下来我们写一个函数来读取图片，两个参数分别是图片地址和图片结构体
```c++
bool image_load_from_file(const std::string & fname, image_u8 & img) {
    int nx, ny, nc;
    auto data = stbi_load(fname.c_str(), &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "%s: failed to load '%s'\n", __func__, fname.c_str());
        return false;
    }

    img.nx = nx;
    img.ny = ny;
    img.data.resize(nx * ny * 3);
    memcpy(img.data.data(), data, nx * ny * 3);

    stbi_image_free(data);

    return true;
}
```

## 运行
首先初始化ggml
```c++
ggml_time_init();
```

接下来加载模型
```c++
mnist_model model;
// load the model
{
    const int64_t t_start_us = ggml_time_us();
    if (!mnist_model_load(argv[1], model)) {
        fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, argv[1]);
        return 1;
    }
    

    const int64_t t_load_us = ggml_time_us() - t_start_us;

    fprintf(stdout, "%s: loaded model in %8.2f ms\n", __func__, t_load_us / 1000.0f);
}
```

接下来读取图片并存储为特定格式
```c++
// read a img from a file

image_u8 img0;
std::string img_path = argv[2];
if (!image_load_from_file(img_path, img0)) {
    fprintf(stderr, "%s: failed to load image from '%s'\n", __func__, img_path.c_str());
    return 1;
}
fprintf(stderr, "%s: loaded image '%s' (%d x %d)\n", __func__, img_path.c_str(), img0.nx, img0.ny);


uint8_t buf[784];

// convert the image to a digit

const int64_t t_start_us = ggml_time_us();

for (int i = 0; i < 784; i++) {
    buf[i] = 255 - img0.data[i * 3];
}

for (int i = 0; i < 784; i++) {
    digit.push_back(buf[i] / 255.0f);
}

const int64_t t_convert_us = ggml_time_us() - t_start_us;

fprintf(stdout, "%s: converted image to digit in %8.2f ms\n", __func__, t_convert_us / 1000.0f);
```

接下来进行推理
```c++
const int prediction = mnist_eval(model, 1, digit, nullptr);
fprintf(stdout, "%s: predicted digit is %d\n", __func__, prediction);
```
最后记得释放内存
```c++
ggml_free(model.ctx);
```
## 使用
在`examples/CMakeLists.txt`最后一行加入`add_subdirectory(mnist-torch)`

然后运行`mkdir build && cd build && cmake .. && make mnist-torch -j8`

最后运行`./mnist-torch /path/to/mnist-ggml-model-f32.gguf /path/to/example.png`

记得把`/path/to/mnist-ggml-model-f32.gguf`和`/path/to/example.png`换成你的路径