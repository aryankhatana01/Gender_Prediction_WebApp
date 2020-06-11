import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
import pickle

df1 = pd.read_csv('https://gist.githubusercontent.com/mbejda/7f86ca901fe41bc14a63/raw/38adb475c14a3f44df9999c1541f3a72f472b30d/Indian-Male-Names.csv')
df2 = pd.read_csv('https://gist.githubusercontent.com/mbejda/9b93c7545c9dd93060bd/raw/b582593330765df3ccaae6f641f8cddc16f1e879/Indian-Female-Names.csv')

df = pd.concat([df1,df2],axis=0)
df = df.drop(['race'],axis=1)

df.gender.replace({'f':0,'m':1},inplace=True)
Xfeatures =df['name']
cv = CountVectorizer()
x = cv.fit_transform(df['name'].values.astype('U'))
y = df.gender

from sklearn.model_selection import train_test_split


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train,y_train)
clf.score(X_test,y_test)

print("Train Accuracy of Model",clf.score(X_train,y_train)*100,"%")
print("Test Accuracy of Model",clf.score(X_test,y_test)*100,"%")



pickle.dump(clf,open('Trained.pkl','wb'))
model = pickle.load(open('Trained.pkl','rb'))

def genderpredictor(a):
    test_name = [a]
    vector = cv.transform(test_name).toarray()
    if model.predict(vector) == 0:
        print("Female")
    else:
        print("Male")

genderpredictor('aryan')


