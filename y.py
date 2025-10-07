import torch
print(torch.version.cuda)
print(torch.cuda.is_available())
import tensorflow as tf
print(tf.test.is_built_with_cuda())
print(tf.config.list_physical_devices('GPU'))
