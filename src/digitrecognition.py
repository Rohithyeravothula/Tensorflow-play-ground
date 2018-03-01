import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data", one_hot=True)


learning_rate = 0.0007
epochs = 10
batch_size = 100

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])


hidden_layer_neurons = 300
output_neurons = 10

W1 = tf.Variable(tf.random_normal([784, hidden_layer_neurons], stddev=0.03), name="W1")
b1 = tf.Variable(tf.random_normal([hidden_layer_neurons]), name="b1")

W2 = tf.Variable(tf.random_normal([hidden_layer_neurons, output_neurons], stddev=0.03), name="W2")
b2 = tf.Variable(tf.random_normal([10]))

hidden_layer = tf.add(tf.matmul(x, W1), b1)
hidden_layer = tf.nn.relu(hidden_layer)

y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_layer, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1 - y_clipped), axis=1))
# cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1-y) * tf.log(1 - y_clipped), axis=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cross_entropy)


init = tf.global_variables_initializer()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess=tf.Session()
sess.run(init)

total_batch = int(len(mnist.train.labels)/batch_size)
for epoch in range(epochs):
    avg_cost = 0
    for i in range(total_batch):
        batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
        _, c = sess.run([optimizer, cross_entropy], feed_dict={x: batch_x, y: batch_y})
        avg_cost += c/total_batch
    print("epoch: {}, avg cost: {}".format(epoch, avg_cost))


accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y:mnist.train.labels})
print("accurary: {}".format(accuracy))


# print(len(mnist.test.images))