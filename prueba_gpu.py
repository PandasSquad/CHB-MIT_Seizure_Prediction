import tensorflow as tf
print("TensorFlow está utilizando GPU:", tf.test.is_built_with_cuda() and tf.test.is_gpu_available())

