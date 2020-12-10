#Import the Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier

import pickle
#Import the dataset
data = pd.read_csv('Loan_Dataset.csv')
#Datatypes in the dataset
print(data.info())
#Fill the missing values with the mean values
data = data.fillna(data.mean().iloc[0])

#Print the numerical and categorical features
numeric_features = data.select_dtypes(include = ['int64', 'float64']).columns
categorical_features = data.iloc[:, 0:12].select_dtypes(include = ['object']).columns
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)

#Replace the values of the categorical features with numerical values
data['Gender'].replace({'Male': 1, 'Female': 0}, inplace=True)
data['Married'].replace({'Yes': 1, 'No': 0}, inplace=True)
data['Education'].replace({'Graduate': 1, 'Not Graduate': 0}, inplace=True)
data['Dependents'].replace({'3+':3},inplace=True)
data['Self_Employed'].replace({'Yes':1,'No':0},inplace=True)
data['Property_Area'].replace({'Urban':2,'Semiurban':1,'Rural':0},inplace=True)
data['Loan_Status'].replace({'Y':'YES','N':'NO'},inplace=True)

#Output parameter
y = data['Loan_Status']
#Input parameters
X = data.drop(['Loan_Status', 'Loan_ID',], axis = 1)
print(data.head())
print(data.shape)
#Split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Initialize CatBoostClassifier
model = CatBoostClassifier(iterations=2, learning_rate=0.2, depth=2)
#Fit the model with the train data
model.fit(X_train, y_train)
#Predict the output of the test data
ypred = model.predict(X_test)
print('\n\nCatBoostClassifier Accuracy:', accuracy_score(y_test, ypred))

pickle.dump(model,open('model.pkl','wb'))


