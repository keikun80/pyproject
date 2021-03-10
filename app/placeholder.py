import os
os.environ['TF_CPP_MIN_LOG_LEVEl']='2' 
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf

tf.compat.v1.disable_eager_execution() 
try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass



node1 = tf.compat.v1.placeholder(tf.float32)
node2 = tf.compat.v1.placeholder(tf.float32)

node3 =  node1 + node2

with tf.compat.v1.Session() as sess:
    print(sess.run(node3, feed_dict={node1:4, node2:9}))
    print(sess.run(node3, feed_dict={node1:[6,1], node2:[4,9]}))

print ("====================")
#print(output)
print ("====================")