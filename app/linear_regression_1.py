import os
os.environ['TF_CPP_MIN_LOG_LEVEl']='2' 
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import tensorflow as tf

tf.compat.v1.disable_eager_execution()  
tf.compat.v1.disable_v2_behavior()

try:
    tf.config.set_visible_devices([], 'GPU')
    visible_devices = tf.config.get_visible_devices()
    for device in visible_devices:
        assert device.device_type != 'GPU'
except:
    pass



x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable (tf.compat.v1.random_normal([1]), name='Weight')
b = tf.Variable (tf.compat.v1.random_normal([1]), name='bias')

hypothesis = x_train * W + b
cost =tf.compat.v1.reduce_mean(tf.compat.v1.square(hypothesis - y_train))

optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for step in range(2001):
        _, cost_val, W_val, b_val = sess.run([train, cost,W,b]) 

        if step % 20 == 0:
            print(step, cost_val, W_val, b_val)
#print ("====================")
#print(output)
#print ("====================")