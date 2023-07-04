import numpy as np
import pandas as pd

class NaiveBayesClassifier:
    def __init__(self):
        self.prior_prob = {}
        self.cond_prob = {}

    def fit(self, X_train, y_train):
        n_samples, n_features = X_train.shape
        self.classes = np.unique(y_train)
        n_classes = len(self.classes)

        # Calculate prior probability for each class
        for c in self.classes:
            self.prior_prob[c] = np.mean(y_train == c)

        # Calculate conditional probability for each feature given each class
        for c in self.classes:
            X_c = X_train[y_train == c]
            self.cond_prob[c] = {}
            for i in range(n_features):
                self.cond_prob[c][i] = {
                    'mean': np.mean(X_c[:, i]),
                    'std': np.std(X_c[:, i])}

    def predict(self, X_test):
        y_pred = []
        for x in X_test:
            probs = {c: self.prior_prob[c] for c in self.classes}
            for c in self.classes:
                for i, x_i in enumerate(x):
                    mean = self.cond_prob[c][i]['mean']
                    std = self.cond_prob[c][i]['std']
                    if std == 0:
                        std = 1e-9
                    probs[c] *= self.gaussian_prob(x_i, mean, std)
            y_pred.append(max(probs, key=probs.get))
        return y_pred

    def gaussian_prob(self, x, mean, std):
        exponent = np.exp(-((x - mean) ** 2 / (2 * std ** 2)))
        return (1 / (np.sqrt(2 * np.pi) * std)) * exponent

    def accuracy(self, X_test, y_test):
        y_pred = self.predict(X_test)
        return np.mean(y_pred == y_test)
    
# Load the dataset
data = pd.read_csv('naivebayes.csv')

# Separate the features and the target variable
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Split the dataset into training and testing sets
n_samples = X.shape[0]
test_size = 0.33
n_test = int(n_samples * test_size)
n_train = n_samples - n_test

# Shuffle the indices
indices = np.random.permutation(n_samples)

# Split the indices into training and testing indices
train_indices = indices[:n_train]
test_indices = indices[n_train:]

# Split the data into training and testing sets
X_train = X[train_indices]
y_train = y[train_indices]
X_test = X[test_indices]
y_test = y[test_indices]

# Train the model
nb = NaiveBayesClassifier()
nb.fit(X_train, y_train)

# Test the model
accuracy = nb.accuracy(X_test, y_test)
print("Accuracy:", accuracy)
