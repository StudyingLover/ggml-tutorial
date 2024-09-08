import sys
import struct
import json
import numpy as np
import re

import gguf

import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

if len(sys.argv) != 2:
    print("Usage: convert-pth-to-ggml.py model\n")
    sys.exit(1)

state_dict_file = sys.argv[1]
fname_out = "model/mnist-cnn-ggml-model-f32.gguf"

state_dict = torch.load(state_dict_file, map_location=torch.device("cpu"))

model = torch.load(state_dict_file, map_location=torch.device("cpu"))
print(model)
gguf_writer = gguf.GGUFWriter(fname_out, "cnn")

conv1_weight = model["conv1.weight"].data.numpy()
conv1_weight = conv1_weight.astype(np.float32)
gguf_writer.add_tensor("conv1_weight", conv1_weight)

conv1_bias = model["conv1.bias"].data.numpy()
conv1_bias = conv1_bias.astype(np.float32)
gguf_writer.add_tensor("conv1_bias", conv1_bias)

conv2_weight = model["conv2.weight"].data.numpy()
conv2_weight = conv2_weight.astype(np.float32)
gguf_writer.add_tensor("conv2_weight", conv2_weight)

conv2_bias = model["conv2.bias"].data.numpy()
conv2_bias = conv2_bias.astype(np.float32)
gguf_writer.add_tensor("conv2_bias", conv2_bias)

fc1_weight = model["fc1.weight"].data.numpy()
fc1_weight = fc1_weight.astype(np.float32)
gguf_writer.add_tensor("fc1_weight", fc1_weight)

fc1_bias = model["fc1.bias"].data.numpy()
fc1_bias = fc1_bias.astype(np.float32)
gguf_writer.add_tensor("fc1_bias", fc1_bias)

fc2_weight = model["fc2.weight"].data.numpy()
fc2_weight = fc2_weight.astype(np.float32)
gguf_writer.add_tensor("fc2_weight", fc2_weight)

fc2_bias = model["fc2.bias"].data.numpy()
fc2_bias = fc2_bias.astype(np.float32)
gguf_writer.add_tensor("fc2_bias", fc2_bias)

gguf_writer.write_header_to_file()
gguf_writer.write_kv_data_to_file()
gguf_writer.write_tensors_to_file()
gguf_writer.close()
print("Model converted and saved to '{}'".format(fname_out))
