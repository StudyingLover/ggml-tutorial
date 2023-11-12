import sys
import numpy as np
import gguf
import torch

if len(sys.argv) != 2:
    print("Usage: convert-pth-to-ggml.py model\n")
    sys.exit(1)

state_dict_file = sys.argv[1]
fname_out = "model/mnist-ggml-model-f32.gguf"

state_dict = torch.load(state_dict_file, map_location=torch.device('cpu'))

model = torch.load(state_dict_file, map_location=torch.device('cpu'))
print(model.keys())
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
print("Model converted and saved to '{}'".format(fname_out))