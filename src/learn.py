import tensorflow as tf


x1 = tf.constant([1,2,3,4])
x2 = tf.constant([3,3,3,3])

result = tf.multiply(x1, x2)

print(result)

config = tf.ConfigProto(log_device_placement=True)

with tf.Session(config=config) as sess:
    result = sess.run(result)
    print(result)