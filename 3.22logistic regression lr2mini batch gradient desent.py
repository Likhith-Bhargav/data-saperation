import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def logistic_regression(X_train, y_train, num_iterations, learning_rate, batch_size):
    w = np.zeros((X_train.shape[1], 1))
    b = 0
    costs = []
    num_batches = int(np.ceil(len(y_train) / batch_size))
    for i in range(min(num_iterations, 100)):
        # shuffle the data before each iteration
        indices = np.random.permutation(len(y_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        for j in range(num_batches):
            # get the current batch
            start_idx = j * batch_size
            end_idx = (j+1) * batch_size
            X_batch = X_train_shuffled[start_idx:end_idx]
            y_batch = y_train_shuffled[start_idx:end_idx]
            # calculate predicted values
            z = np.dot(X_batch, w) + b
            y_pred = 1 / (1 + np.exp(-z))
            # calculate cost function
            cost = -np.mean(y_batch*np.log(y_pred.reshape(-1, 1)) + (1-y_batch)*np.log(1-y_pred.reshape(-1, 1)))
            costs.append(cost)
            # calculate gradients
            dw = np.dot(X_batch.T, (y_pred - y_batch)) / len(y_batch)
            db = np.sum(y_pred - y_batch) / len(y_batch)
            # update weights and bias
            w -= learning_rate * dw
            b -= learning_rate * db
    return w, b, costs

# Define prediction function
def predict(X, w, b, threshold=0.5):
    z = np.dot(X, w) + b
    y_pred = 1 / (1 + np.exp(-z))
    y_pred_class = (y_pred >= threshold).astype(int)
    return y_pred_class.reshape(-1, 1)

# Load data
data = pd.read_csv("sdata.csv")
# Split data into training and testing sets
train_size = int(0.67 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]
# Prepare data for training and testing
X_train = train_data.drop('target_variable', axis=1).values
y_train = train_data['target_variable'].values.reshape(-1, 1)
X_test = test_data.drop('target_variable', axis=1).values
y_test= test_data['target_variable'].values.reshape(-1, 1)
# Train the model and get costs
w, b, costs = logistic_regression(X_train, y_train, num_iterations=1000, learning_rate=0.01, batch_size=32)
# Make predictions on the test data
y_pred = predict(X_test, w, b, threshold=0.5)
# Calculate accuracy
accuracy = np.mean(y_pred == y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))
# Calculate precision and recall
tp = np.sum((y_test == 1) & (y_pred == 1))
fp = np.sum((y_test == 0) & (y_pred == 1))
tn = np.sum((y_test == 0) & (y_pred == 0))
fn = np.sum((y_test == 1) & (y_pred == 0))
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))

# Plot learning curve
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Learning Curve")
plt.show()