# ggml-tutorial
ggml是一个非常轻量级的机器学习推理框架，但是我并没有找到相关的教程，只有官方的一个仓库[ggml](https://github.com/ggerganov/ggml)还有一些使用了ggml的项目[llama.cpp](https://github.com/ggerganov/llama.cpp),[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp),[chatglm.cpp](https://github.com/li-plus/chatglm.cpp)等等，所以我萌生了创建这样一个仓库的想法，记录一下学习ggml的过程，同时作为一个ggml的教程，希望能够帮助到一些人。

## 如果使用
very easy,but maybe a little a bit silly

把这个仓库下所有内容复制到ggml的example仓库下，然后在`examples/CMakeLists.txt`写入每个例子的名字，在每个例子的README里面我有详细说明。

## TODO
- [x] mnist-torch
- [ ] mnist-torch-conv
- [ ] mnist-torch-attention
- [ ] yolov5
- [ ] 还没想好~~~