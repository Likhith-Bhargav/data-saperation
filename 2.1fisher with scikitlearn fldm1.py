import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt

#this is used to read the data from the file sdata.csv by droping the nan values
data = pd.read_csv("fisherdata.csv").dropna()

#these lines split the data into training and testing by taking 67% of data as training and remaining as testing 
# which is taken according to question requirements  
X_train, X_test, y_train, y_test = train_test_split(data.iloc[:,:-1], data.iloc[:,-1], test_size=0.33, random_state=42)

# we claculate the mean and variance of both postitive(1) and negative classes(0)
X_train_pos = X_train[y_train == 1]
X_train_neg = X_train[y_train == 0]
mean_pos = X_train_pos.mean(axis=0)
mean_neg = X_train_neg.mean(axis=0)
var_pos = X_train_pos.var(axis=0)  
var_neg = X_train_neg.var(axis=0)

# in fisher linear discriminant analysis we compute sb and sw as 
#sb= (m2-m1).(m2-m1)^t  where m2 is mean of positive class and m1 is mean of negative class and we cmpute the sw from their variances
Sb = np.outer(mean_pos - mean_neg, mean_pos - mean_neg)
Sw = np.diag(var_pos) + np.diag(var_neg)
#we obtain our projection vector by getting the largest eigen vector of the product (sw)^-1(sb)
eigvals, eigvecs = np.linalg.eig(np.linalg.inv(Sw) @ Sb)
opt_proj_dir = eigvecs[:, np.argmax(eigvals)]

# Project the data onto the optimal projection direction
X_train_proj = X_train @ opt_proj_dir
X_test_proj = X_test @ opt_proj_dir

# gaussianNB is a class implemented in scikitlearn which trains model according to training data
gnb = GaussianNB()
gnb.fit(X_train_proj.to_numpy().reshape(-1, 1).real, y_train.to_numpy())

# Find the decision boundary
db = np.dot(opt_proj_dir, (mean_pos + mean_neg) / 2)

# it reshapes the array into single column and takes the real values  
#score is basically defined in scikitlearn which gives the accuracy by checking X_test_proj_np and comparing with y_test (specific to our code)
X_test_proj_np = X_test_proj.to_numpy().reshape(-1, 1).real
accuracy = gnb.score(X_test_proj_np, y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Convert X_train_proj to a NumPy array and extract the first column
X_train_proj_np = X_train_proj.values.reshape(-1, 1).real

# Plot the data and the decision boundary
#this is for positive points
if (X_train_proj_np[y_train == 1].size > 0):
#this is for negative points 
 plt.scatter(X_train_proj_np[y_train == 1], y_train[y_train == 1], c='red', label='Positive')
if (X_train_proj_np[y_train == 0].size > 0):
 plt.scatter(X_train_proj_np[y_train == 0], y_train[y_train == 0], c='blue', label='Negative')
#this is for desicion boundary
plt.axvline(x=db, linestyle='--', color='black')
#labeling x-axis
plt.xlabel('Projected Feature')
#labeling y-axis
plt.ylabel('Class Label')
plt.title('FLDA Decision Boundary')
plt.legend()
plt.show()
