import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
pd_df = pd.read_csv('my_poses/csvs_out_basic.csv')
print(pd_df.head())
values = pd_df.iloc[:, 2:].to_numpy()
print(values)
labels = pd_df.iloc[:, 1]
print(labels)
x_train, x_test, y_train, y_test = train_test_split(values, labels , test_size=0.2, random_state=42)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(x_train, y_train)
prediction = []
for i in range(30):
    p = knn.predict(x_test[i].reshape(1,-1))
    prediction.append(p[0])
print((y_test[:30] == prediction).sum()/len(prediction))


import pickle
knnPickle = open('poses_knn', 'wb')
pickle.dump(knn, knnPickle)


loaded_model = pickle.load(open('poses_knn', 'rb'))
prediction = []
for i in range(30):
    p = loaded_model.predict(x_test[i].reshape(1,-1))

    prediction.append(p[0])
print((y_test[:30] == prediction).sum()/len(prediction))

# result = loaded_model.predict(X_test)