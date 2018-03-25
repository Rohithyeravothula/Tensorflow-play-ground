import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


columns = ["Pclass","Sex", "Age","SibSp","Parch","Fare"]


def one_hot_encoder(labels):
    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels, n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode



def read_file(filename):
    data_full = pd.read_csv(filename)
    data=data_full
    if "Survived" in list(data.columns.values):
        y = data["Survived"]
    else:
        y = None
    data["Sex"] = data["Sex"].apply(lambda x: 1 if x=="male" else 0)
    data = data[columns]
    for col in columns:
        data[col].fillna(0, inplace=True)
        if col != "Sex":
            data[col] = (data[col] - data[col].mean())/data[col].std()
    if y is not None:
        encoder = LabelEncoder()
        encoder.fit(y)
        y = encoder.transform(y)
        Y = one_hot_encoder(y)
    else:
        Y = None
    return data, Y, data_full


filename = "../../data/titanic/train.csv"
X, Y, _ = read_file(filename)
X, Y = shuffle(X, Y, random_state=1)
train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=100)


hidden_layer_neurons = 5
hidden_layer2_neurons = 3
output_layer_neurons = 2
learning_rate = 0.7
epochs = 20000

x = tf.placeholder(tf.float32, [None, len(columns)])
y = tf.placeholder(tf.float32, [None, output_layer_neurons])

W1 = tf.Variable(tf.truncated_normal([len(columns), hidden_layer_neurons], stddev=0.03))
b1 = tf.Variable(tf.truncated_normal([hidden_layer_neurons]))

W3 = tf.Variable(tf.truncated_normal([hidden_layer_neurons, hidden_layer2_neurons], stddev=0.08))
b3 = tf.Variable(tf.truncated_normal([hidden_layer2_neurons]))

W2 = tf.Variable(tf.truncated_normal([hidden_layer2_neurons, output_layer_neurons]))
b2 = tf.Variable(tf.truncated_normal([output_layer_neurons]))

hidden_layer = tf.add(tf.matmul(x, W1), b1)
hidden_layer = tf.nn.sigmoid(hidden_layer)

hidden_layer2 = tf.add(tf.matmul(hidden_layer, W3), b3)
hidden_layer2 = tf.nn.sigmoid(hidden_layer2)

output_layer = tf.add(tf.matmul(hidden_layer2, W2), b2)
output_layer = tf.nn.sigmoid(output_layer)


cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(output_layer) + (1-y) * tf.log(1 - output_layer), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)


init = tf.global_variables_initializer()
sess = tf.Session()

sess.run(init)



for epoch in range(epochs):
    sess.run(optimizer, feed_dict={x: train_x, y: train_y})
    cost = sess.run(cross_entropy, feed_dict={x: train_x, y: train_y})
    # print(cost)
    if epoch%100 == 0:
        print(epoch, cost)




prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output_layer, 1))
accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))
accuracy = sess.run(accuracy, feed_dict={x: test_x, y: test_y})
print(accuracy)



model_test = "../../data/titanic/test.csv"
X,Y, dtf = read_file(model_test)

output = tf.argmax(output_layer, 1)
output_val = list(sess.run(output, feed_dict={x: X}))

pids = list(dtf["PassengerId"].values)
l = len(output_val)
s=""
for p in range(0, l):
    s+="{},{}\n".format(pids[p], output_val[p])
f=open("result.txt", 'w+')
f.write(s)
f.close()
