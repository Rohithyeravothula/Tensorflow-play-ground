import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np


def one_hot_encoder(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode


def read_dataset(input_file):
    df = pd.read_csv(input_file, header=None)

    X = df[df.columns[0:60]]
    y = df[df.columns[60]]
    # print(y)

    # encoding => vectorizing the output

    encoder = LabelEncoder()
    encoder.fit(y)

    y = encoder.transform(y)
    Y = one_hot_encoder(y)
    return (X, Y)



filename = "../data/sonar/all.txt"
X, Y = read_dataset(filename)

X, Y = shuffle(X, Y, random_state=1)

train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=415)



# paramenters
learning_rate = 0.03
# training_epochs = 300
cost_history = np.empty(shape=[1], dtype=float)
n_dim = X.shape[1]
n_classes = 2
model_path = "my_model"

# print(n_dim, cost_history)


n_hidden1 = 60
n_hidden2 = 30
n_hidden3 = 30
n_hidden4 = 30



x = tf.placeholder(tf.float32, [None, n_dim])
W = tf.Variable(tf.zeros([n_dim, n_classes]))
b = tf.Variable(tf.zeros([n_classes]))
y_ = tf.placeholder(tf.float32, [None, n_classes])


def multilayer_perceptron(x, weights, biases):

    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.sigmoid(layer_1)

    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.sigmoid(layer_2)

    layer_3 = tf.add(tf.matmul(layer_2, weights['h3']), biases['b3'])
    layer_3 = tf.nn.sigmoid(layer_3)

    layer_4 = tf.add(tf.matmul(layer_3, weights['h4']), biases['b4'])
    layer_4 = tf.nn.relu(layer_4)

    out_layer = tf.matmul(layer_4, weights['out']) + biases['out']
    return out_layer


weight = {
    'h1': tf.Variable(tf.truncated_normal([n_dim, n_hidden1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden1, n_hidden2])),
    'h3': tf.Variable(tf.truncated_normal([n_hidden2, n_hidden3])),
    'h4': tf.Variable(tf.truncated_normal([n_hidden3, n_hidden4])),
    'out': tf.Variable(tf.truncated_normal([n_hidden4, n_classes]))
}

biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden1])),
    'b2': tf.Variable(tf.truncated_normal([n_hidden2])),
    'b3': tf.Variable(tf.truncated_normal([n_hidden3])),
    'b4': tf.Variable(tf.truncated_normal([n_hidden4])),
    'out': tf.Variable(tf.truncated_normal([n_classes]))
}


print(weight['h1'].dtype, type(test_x))


# Initialize variables

init = tf.global_variables_initializer()

saver = tf.train.Saver()

y = multilayer_perceptron(x, weight, biases)


# cost function and optimizers

# what is sigmoid and why use it?
cost_function = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
training_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost_function)


sess = tf.Session()
sess.run(init)


mse_history = []
accuracy_history = []

# cost = sess.run(cost_function, feed_dict={x: train_x, y_: train_y})
# print(cost)
training_epochs = 100
for epoch in range(training_epochs):
    sess.run(training_step, feed_dict={x:train_x, y_:train_y})
    cost = sess.run(cost_function, feed_dict={x:train_x, y_:train_y})
    # print(cost)
    cost_history = np.append(cost_history, cost)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    pred_y = sess.run(y, feed_dict={x:test_x})
    mse = tf.reduce_mean(tf.square(pred_y - test_y))
    mse_val = sess.run(mse)
    mse_history.append(mse_val)
    accuracy = (sess.run(accuracy, feed_dict={x:train_x, y_: train_y}))
    accuracy_history.append(accuracy)
    print("epoch {}, cost {}, mse {}, train_acc {}".format(epoch, cost, mse_val, accuracy))

# saver.save(sess, model_path)

#
# plt.plot(mse_history, 'r')
# plt.show()
# plt.plot(accuracy_history)
# plt.show()
#
#
#
#
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
acc_val = sess.run(accuracy, feed_dict={x: test_x, y_:test_y})
print("accuracy: {}".format(acc_val))
#
pred_y = sess.run(y, feed_dict={x: test_x})
mse = tf.reduce_mean(tf.square(pred_y - test_y))
print("MSE {}".format(sess.run(mse)))
