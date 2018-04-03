# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())


# import tensorflow as tf
# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
