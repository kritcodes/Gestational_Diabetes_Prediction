import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

data = pd.read_csv('datasets_diabetes.csv')

data.head()

plt.figure(figsize=(10,7))
sns.barplot(x='Outcome', y='Age', data=data,errwidth=5)

from sklearn.model_selection import  train_test_split

data.columns

x = data.drop('Outcome',axis=1)
y = data['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=101)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(x_train,y_train)

pickle.dump(knn, open('model.pkl','wb'))

model = pickle.load(open('model.pkl','rb'))

preg = input('Enter the Pregnancy month: ')
glucose = input('Enter the Glucose amount: ')
bloodpres = input('Enter the Blood Pressure level: ')
skinthick = input('Enter the Skin Thickness: ')
insulin = input ('Enter the skin thickness: ')
bmi = input ('Enter the  BMI: ')
dpf = input ('Enter the Diabetes Pedigree Function: ')

pred = knn.predict([[preg,glucose,bloodpres,skinthick,insulin,bmi,dpf]])

if pred == 1:
    print ('The person is Diagnosed Diabetes')
    
else:
    print('The person is not Diagnosed Diabetes')