import numpy as np 
from flask import Flask, request, render_template, jsonify
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
app = Flask(__name__)
import pandas as pd
model = pickle.load(open('Trained.pkl', 'rb'))



@app.route('/')
def home():
	return render_template('index.html')

@app.route('/Predict', methods=['Post'])
def predict():
	df1 = pd.read_csv('https://gist.githubusercontent.com/mbejda/7f86ca901fe41bc14a63/raw/38adb475c14a3f44df9999c1541f3a72f472b30d/Indian-Male-Names.csv')
	df2 = pd.read_csv('https://gist.githubusercontent.com/mbejda/9b93c7545c9dd93060bd/raw/b582593330765df3ccaae6f641f8cddc16f1e879/Indian-Female-Names.csv')

	df = pd.concat([df1,df2],axis=0)
	df = df.drop(['race'],axis=1)

	cv = CountVectorizer()
	x = cv.fit_transform(df['name'].values.astype('U'))
	y = df.gender
	namequery = request.form['Name']
	data = [namequery]
	vect = cv.transform(data).toarray()
	prediction = model.predict(vect)
	if prediction == 0:
		predicted = 'Female'
	else:
		predicted = 'Male'

	return render_template('index.html', prediction_gen = '{} should be a : {}'.format(namequery,predicted))


if __name__ == '__main__':
	app.run(debug=True)
