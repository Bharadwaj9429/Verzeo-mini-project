### importing the packages
import pandas as pd
import numpy as np

### import the dataset
df = pd.read_csv("D:/Iris (1).csv")
df.head()

#### EDA
df.columns
df1 = df.describe()

############# Method 1 Using Decision Tree Classification Model
df['Species'].unique()
df['Species'].value_counts()
colnames = list(df.columns)

predictors = colnames[:4]
target = colnames[5]

# Splitting data into training and testing data set
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size = 0.2)

from sklearn.tree import DecisionTreeClassifier as DT


model = DT(criterion = 'entropy')
model.fit(train[predictors], train[target])


# Prediction on Test Data
preds = model.predict(test[predictors])
pd.crosstab(test[target], preds, rownames=['Actual'], colnames=['Predictions'])

np.mean(preds == test[target]) # Test Data Accuracy 

# Prediction on Train Data
preds = model.predict(train[predictors])
pd.crosstab(train[target], preds, rownames = ['Actual'], colnames = ['Predictions'])

np.mean(preds == train[target]) # Train Data Accuracy

############### Method 2 Using SVM Technique
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

train,test = train_test_split(df, test_size = 0.20)

train_X = train.iloc[:, :5]
train_y = train.iloc[:, 5]
test_X  = test.iloc[:, :5]
test_y  = test.iloc[:, 5]

##### Using linear model SVC
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X, train_y)
pred_test_linear = model_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

pred_train_linear = model_linear.predict(train_X)

np.mean(pred_train_linear == train_y)

# kernel = rbf
model_rbf = SVC(kernel = "rbf")
model_rbf.fit(train_X, train_y)
pred_test_rbf = model_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)

pred_train_rbf = model_rbf.predict(train_X)

np.mean(pred_train_rbf==train_y)

#kernel = poly
model_poly = SVC(kernel = "poly")
model_poly.fit(train_X, train_y)
pred_test_poly = model_poly.predict(test_X)

np.mean(pred_test_poly==test_y)

pred_train_poly = model_poly.predict(train_X)

np.mean(pred_train_poly==train_y)

#kernel = sigmoid
model_sigmoid = SVC(kernel = "sigmoid")
model_sigmoid.fit(train_X, train_y)
pred_test_sigmoid = model_sigmoid.predict(test_X)

np.mean(pred_test_sigmoid==test_y)

pred_train_sigmoid = model_sigmoid.predict(train_X)

np.mean(pred_train_sigmoid==train_y)


#### By comparing these two models, we can say that the Classification models i.e., SVM & Decision Tree Classification models
#### are getting overfitting , so we can say that.....
###### Decision Tree : Train Accuracy : 1.0 & Test Accuracy : 1.0
###### SVM : Kernel : "Linear" : Train Accuracy : 1.0 & Test Accuracy : 1.0
###### SVM : Kernel : "RBF" : Train Accuracy : 1.0 & Test Accuracy : 0.99
###### SVM : Kernel : "Poly" : Train Accuracy : 0.97 & Test Accuracy : 0.96
###### SVM : Kernel : "Sigmoid" : Train Accuracy : 0.33 & Test Accuracy : 0.23
