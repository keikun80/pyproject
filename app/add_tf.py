import os
os.environ['TF_CPP_MIN_LOG_LEVEl']='2' 
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf

try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass



node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)

@tf.function
def forward():
    return node1 + node2

output = forward() 

print ("====================")
print(output)
print ("====================")