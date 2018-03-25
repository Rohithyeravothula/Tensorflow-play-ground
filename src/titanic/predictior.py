import pandas as pd
from sklearn import preprocessing

columns = ["Pclass","Sex", "Age","SibSp","Parch","Fare"]


data = pd.read_csv("test.csv")
data = data[columns]
for col in data:
    data[col].fillna(0, inplace=True)

data["Sex"] = data["Sex"].apply(lambda x: 1 if x=="male" else 0)

for col in columns:
    data[col] = (data[col] - data[col].mean())/data[col].std()
print(data)