import tensorflow as tf

hello = tf.constant("Hello ");

@tf.function
def forward():
    return hello

out_a = forward()
print(out_a)