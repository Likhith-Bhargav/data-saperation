import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Define logistic regression function
#num_iterations : number of iterations for gradient desent algorithm to run
#learning_rate : detrmine step size for desent
def logistic_regression_sgd(X_train, y_train, num_iterations, learning_rate):
    w = np.zeros((X_train.shape[1], 1))
    b = 0
    costs = []
    for i in range(num_iterations):
        for j in range(X_train.shape[0]):
            z = np.dot(X_train[j], w) + b
            y_pred = 1 / (1 + np.exp(-z))
            cost = -y_train[j]*np.log(y_pred) - (1-y_train[j])*np.log(1-y_pred)
            dw = X_train[j].reshape(-1, 1) * (y_pred - y_train[j])
            db = y_pred - y_train[j]
            w -= learning_rate * dw
            b -= learning_rate * db
        costs.append(np.mean(cost))
    return w, b, costs

# Define prediction function
def predict(X, w, b, threshold=0.5):
    z = np.dot(X, w) + b
    y_pred = 1 / (1 + np.exp(-z))
    y_pred_class = (y_pred >= threshold).astype(int)
    return y_pred_class.reshape(-1, 1)

#this is used to read the data from the file sdata.csv
data = pd.read_csv("sdata.csv")

#these lines split the data into training and testing by taking 67% of data as training and remaining as testing 
# which is taken according to question requirements 
train_size = int(0.67 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Prepare data for training and testing 
#X_train takes the values of attributes 
# y_train trains the model either 0 or 1 according to corresponding attributes 
#similarly it also tests x and y accordingly
X_train = train_data.drop('target_variable', axis=1).values
y_train = train_data['target_variable'].values.reshape(-1, 1)
X_test = test_data.drop('target_variable', axis=1).values
y_test= test_data['target_variable'].values.reshape(-1, 1)

# Train the model with SGD and get costs
w_sgd, b_sgd, costs_sgd = logistic_regression_sgd(X_train, y_train, num_iterations=1000, learning_rate=0.01)

# Make predictions on the test data
y_pred_sgd = predict(X_test, w_sgd, b_sgd, threshold=0.5)

# Calculate accuracy, precision, and recall
accuracy_sgd = np.mean(y_pred_sgd == y_test)


# Plot learning curve
plt.plot(costs_sgd)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Learning Curve (SGD)")
plt.show()

# Print results
print('Accuracy (SGD): {:.2f}%'.format(accuracy_sgd * 100))
