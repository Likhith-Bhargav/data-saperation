import numpy as np
import pandas as pd
import random
#this is used to read the data from the file data.csv
data = pd.read_csv('data.csv')

#these lines split the data into training and testing by taking 67% of data as training and remaining as testing 
# which is taken according to question requirements  
training_data = data.sample(frac=0.67, random_state=1)
testing_data = data.drop(training_data.index)

#defining the perceptron algorithm
class PerceptronModel():
    def __init__(self, n_inputs):
        self.weights = [float(random.uniform(-1,1)) for _ in range(n_inputs)]
        self.bias = float(random.uniform(-1,1))
#It first initializes the activation to 0.0, then multiplies each input by its corresponding weight and adds the result to the activation. The bias is then added to the activation.
    def activation(self, inputs):
        activation = 0.0
        for i in range(len(inputs)):
            activation += self.weights[i] * float(inputs[i])
        activation += self.bias
        return 1.0 if activation >= 0.0 else 0.0
#it trains the data by running all through the data once and calculates the error between present value and update the bial accordingly
    def train(self, dataset, n_epochs, learning_rate):
        for epoch in range(n_epochs):
            sum_error = 0.0
            for inputs, target in dataset:
                prediction = self.activation(inputs)
                error = target - prediction
                sum_error += error**2
                self.bias += learning_rate * error
                for i in range(len(inputs)):
                    self.weights[i] += learning_rate * error * float(inputs[i])
            if sum_error == 0.0:
                print('Epoch {} converged'.format(epoch))
                break
#it calculates the number of inputs of dataset by excluding the last target_variable column
n_inputs = len(training_data.columns) - 1
#creats an instance for class perceptron
model = PerceptronModel(n_inputs)
#creates the list according to each column
dataset = [(row[:-1], row[-1]) for _, row in training_data.iterrows()]
#this determines how many times to pass through complete data
n_epochs = 100
learning_rate = 0.1
#used to train the data 
model.train(dataset, n_epochs, learning_rate)
#helps in testing the data 
#in our case we took 33% of data for testing 
#while testing if we get correct output as target_variabe then we update the correct variable with 1
correct = 0
for _, row in testing_data.iterrows():
    inputs, target = row[:-1], row[-1]
    prediction = model.activation(inputs)
    if target == prediction:
        correct += 1
#calculates the accuracy depending on correct variable        
accuracy = correct / len(testing_data)
print('Accuracy: {:.2f}%'.format(accuracy * 100))
