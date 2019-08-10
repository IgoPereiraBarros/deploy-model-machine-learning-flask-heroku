# https://towardsdatascience.com/designing-a-machine-learning-model-and-deploying-it-using-flask-on-heroku-9558ce6bde7b

import os
import numpy as np
import pickle
from flask import Flask, render_template, request


def value_prediction(to_predict_list):
	to_predict = np.array(to_predict_list).reshape(1, 12)
	loaded_model = pickle.load(open('models/decisiontreeclassifier.pkl', 'rb'))
	result = loaded_model.predict(to_predict)
	return result[0]


#creating instance of the class
app = Flask(__name__)

#to tell flask what url shoud trigger the function index()
@app.route('/')
@app.route('/index')
def index():
	return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
	if request.method == 'POST':
		to_predict_list = request.form.to_dict()
		to_predict_list = list(to_predict_list.values())
		to_predict_list = list(map(int, to_predict_list))
		result = value_prediction(to_predict_list)

		if int(result) == 1:
			prediction = 'Income more than 50K'
		else:
			prediction = 'Income less than 50K'

		return render_template('result.html', prediction=prediction)


if __name__ == '__main__':
	app.run(debug=False, port=5000)