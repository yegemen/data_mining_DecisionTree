import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('cars.csv')
print(dataset)

x = dataset.iloc[:, [0,1,2,3,4,5,6]].values
y = dataset.iloc[:, 7].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

from sklearn.tree import DecisionTreeClassifier 
model = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

print(pd.Series(y[0:66],y_pred))

yeniOrnek = [[17,8,345,95,2600,16,1975]]
yeniOrnek = sc_x.transform(yeniOrnek)
sonuc = model.predict(yeniOrnek)

print(sonuc)