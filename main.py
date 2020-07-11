import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics

# Get data from CSV
df = pd.read_csv("./data/pima-data.csv")

# We don't need the 'skin' column, since it has full (1.0) correlation with the 'thickness' column
del df['skin']

# Map True and False to 1 and 0
diabetes_map = {True: 1, False: 0}
df['diabetes'] = df['diabetes'].map(diabetes_map)

# Split the data
feature_col_names = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age']
predicted_class_names = ['diabetes']

X = df[feature_col_names].values
Y = df[predicted_class_names].values

split_test_size = 0.30

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=split_test_size, random_state=42)

# Impute missing data
fill_0 = SimpleImputer(missing_values=0, strategy="mean")
X_train = fill_0.fit_transform(X_train)
X_test = fill_0.fit_transform(X_test)

# Training using algorithm - Logistic Regression Cross-Validation
from sklearn.linear_model import LogisticRegressionCV

lr_model = LogisticRegressionCV(n_jobs=-1, Cs=50, cv=100, refit=False, random_state=45, max_iter=100000, class_weight="balanced")
lr_model.fit(X_train, Y_train.ravel())

lr_predict_train = lr_model.predict(X_train)
lr_predict_test = lr_model.predict(X_test)

# Evaluating the model
br = metrics.classification_report(Y_test, lr_predict_test)
print(br)

#End the program
print("END")