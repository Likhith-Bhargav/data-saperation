#accuracy is not good because normalizind the data is not done
#due to unnormalized data we are encountering overflow
#in lr2 we use the normalized data to get good accuracy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define logistic regression function
#num_iterations : number of iterations for gradient desent algorithm to run
#learning_rate : detrmine step size for desent
def logistic_regression(X_train, y_train, num_iterations, learning_rate):
    #first we Initialize weights and bias to 0
    w = np.zeros((X_train.shape[1], 1))
    b = 0
    
    # Initialize list to store costs
    costs = []
    
    # Gradient descent loop w(k+1) <- w(k)-n(d(e(w))/dw)
    for i in range(num_iterations):
        # Calculate predicted values
        z = np.dot(X_train, w) + b
        z = np.clip(z, -500, 500)  # clip values to prevent overflow or underflow
    #sigmiod function(which ranges 0 to 1)  
        y_pred = 1 / (1 + np.exp(-z))

        # Calculate negative log loss cost function(cost function)
        cost = -np.mean(y_train*np.log(y_pred.reshape(-1, 1)) + (1-y_train)*np.log(1-y_pred.reshape(-1, 1)))
        costs.append(cost)
        
        # Calculate gradients at present value of w
        dw = np.dot(X_train.T, (y_pred - y_train)) / len(y_train)
        db = np.sum(y_pred - y_train) / len(y_train)
        
        # Update weights and bias according to the gradient we obtained
        w -= learning_rate * dw
        b -= learning_rate * db
        
    return w, b, costs

# Define prediction function
def predict(X, w, b, threshold=0.5):
    z = np.dot(X, w) + b
    y_pred = 1 / (1 + np.exp(-z))
    y_pred_class = (y_pred >= threshold).astype(int)
    return y_pred_class.reshape(-1, 1)

#this is used to read the data from the file fisherdata.csv
data = pd.read_csv("fisherdata.csv")

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

# Train the model and get costs
w, b, costs = logistic_regression(X_train, y_train, num_iterations=1000, learning_rate=0.01)

# Make predictions on the test data by taking the testing samples from data 
#threshold 0.5 implies that if the result obtained is greater than 0.5 will be classified as positive and remaining as negative

y_pred = predict(X_test, w, b, threshold=0.5)
y_pred = y_pred.reshape(-1, 1)

# Calculate accuracy by checking the obtained y_pred with y_train
accuracy = np.mean(y_pred == y_test)
print('Accuracy: {:.2f}%'.format(accuracy * 100))

# Define function to calculate precision and recall
def precision_recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return precision, recall

# Calculate precision and recall
precision, recall = precision_recall(y_test, y_pred)
print('Precision: {:.2f}'.format(precision))
print('Recall: {:.2f}'.format(recall))


# Plot learning curve
plt.plot(costs)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Learning Curve")
plt.show()
