import tensorflow as tf
import pandas as pd

input = "Titanic/train.csv"

# df = pd.read_csv(input)

# survived = tf.placeholder(tf.float32)

# data = {survived:df["Survived"].apply(lambda x: float(x))}

# print(sess.run(survived, feed_dict=data))


def sex_to_number(inp):
    if inp.lower() == "male":
        return 0.0
    return 1.0



df = pd.read_csv(input)

# variables
survived = tf.placeholder(tf.float32)
sex = tf.placeholder(tf.float32)
age = tf.placeholder(tf.float32)
sibSp = tf.placeholder(tf.float32)
parch = tf.placeholder(tf.float32)
fare = tf.placeholder(tf.float32)
# cabin = tf.placeholder(tf.float32)

headers = ["Survived","Sex","Age","SibSp","Parch","Fare"]

def normalize(dataframe):
    return (dataframe - dataframe.mean())/dataframe.std()

columns = pd.read_csv(input)
data = {survived: normalize(columns["Survived"].fillna(0).apply(lambda x: float(x))),
        sex:normalize(columns["Sex"].apply(sex_to_number).fillna(0).apply(lambda x: float(x))),
        age:normalize(columns["Age"].fillna(0).apply(lambda x: float(x))),
        sibSp:normalize(columns["SibSp"].fillna(0).apply(lambda x: float(x))),
        parch:normalize(columns["Parch"].fillna(0).apply(lambda x: float(x))),
        fare:normalize(columns["Fare"].fillna(0).apply(lambda x: float(x)))}

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


# for ele in df["SibSp"].tolist():
#     if not isinstance(ele, (int, float)):
#         print(ele)
#
# print(sess.run(survived, feed_dict={survived:df["Survived"].apply(lambda x: float(x))}))
# sex_data = df["Sex"].apply(sex_to_number)
# print(type(df["Survived"].apply(lambda x: float(x))))
# print(type(sex_data))
# print(sess.run(age, feed_dict={age:df["Age"].fillna(0).apply(lambda x: float(x))}))
# print(sess.run(sex, feed_dict={sex:df["Sex"].apply(sex_to_number).fillna(0).apply(lambda x: float(x))}))
# print(sess.run(sibSp, feed_dict={sibSp:df["SibSp"].fillna(0).apply(lambda x: float(x))}))
# print(sess.run(parch, feed_dict={parch:df["Parch"].fillna(0).apply(lambda x: float(x))}))
# print(sess.run(fare, feed_dict={fare:df["Fare"].fillna(0).apply(lambda x: float(x))}))
# print(sess.run(survived, feed_dict={sex:}))

# print(df["SibSp"].size)
# print(df["SibSp"].dropna().size)

# weights

w1 = tf.Variable([1.0], tf.float32)
w2 = tf.Variable([1.0], tf.float32)
w3 = tf.Variable([1.0], tf.float32)
w4 = tf.Variable([1.0], tf.float32)
# w5 = tf.Variable([1], tf.float32)
w6 = tf.Variable([1.0], tf.float32)
w7 = tf.Variable([1.0], tf.float32)

# model

predicted = tf.multiply(w2, age) + tf.multiply(w3, sex) + tf.multiply(w4, fare) + w6


# print(sess.run(predicted, feed_dict=data))

# error
squared_error = tf.square(predicted - survived)
loss = tf.reduce_sum(squared_error)

# train
optimizer = tf.train.GradientDescentOptimizer(0.000001)
train = optimizer.minimize(loss)



init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


print(data[age])

# for i in range(0, 100):
#     sess.run(train, feed_dict=data)
#     print(sess.run(loss, feed_dict=data))



# print(df["Pclass"].tolist())


