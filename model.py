import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


# Load dataset
df = pd.read_csv('dataset/adult.csv')

# filling missing values
col_names = df.columns
for c in col_names:
	df[c] = df[c].replace('?', np.NaN)

df = df.apply(lambda x: x.fillna(x.value_counts().index[0]))

#discretisation
df.replace(['Divorced', 'Married-AF-spouse', 
              'Married-civ-spouse', 'Married-spouse-absent', 
              'Never-married','Separated','Widowed'],
             ['divorced','married','married','married',
              'not married','not married','not married'], inplace=True)

#label Encoder
category_col = ['workclass', 'race', 'education','marital-status', 'occupation',
               'relationship', 'sex', 'native-country', 'class']

labelencoder = LabelEncoder()

# creating a map of all the numerical values of each categorical labels.
mapping_dict = {}
for col in category_col:
	df[col] = labelencoder.fit_transform(df[col])
	le_name_mapping = dict(zip(labelencoder.classes_, labelencoder.transform(labelencoder.classes_)))
	mapping_dict[col] = le_name_mapping

print(mapping_dict)

#droping redundant columns
df = df.drop(['fnlwgt','education-num'], axis=1)

# Features and targert
X = df.iloc[:, :12]
y = df.iloc[:, -1]

# Splitting the data in Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)

# Fitting the model
classifier = DecisionTreeClassifier(criterion='gini', random_state=100,
									max_depth=5, min_samples_leaf=5)
classifier.fit(X_train, y_train)

# Prediting the result
y_pred = classifier.predict(X_test)

print ("Desicion Tree using Gini Index\nAccuracy is ", accuracy_score(y_test, y_pred) * 100)

#creating and training a model
#serializing our model to a file called model.pkl
pickle.dump(classifier, open('models/decisiontreeclassifier.pkl', 'wb'))