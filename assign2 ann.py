import tensorflow as tf
from tensorflow import keras

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0
X_test = X_test / 255.0
def build_model(num_hidden_layers, num_neurons, activation_func, optimizer):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28)))
    
    for i in range(num_hidden_layers):
        model.add(keras.layers.Dense(num_neurons, activation=activation_func))
    
    model.add(keras.layers.Dense(10, activation='softmax'))
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
num_epochs = 10

for i in range(1, 16):
    if i <= 9:
        num_hidden_layers = 2
        num_neurons = 100 if i % 3 == 1 else 150
        activation_func = 'tanh' if i % 3 == 1 else ('sigmoid' if i % 3 == 2 else 'relu')
        optimizer = 'sgd'
    elif i <= 12:
        num_hidden_layers = 3
        num_neurons = 100 if i % 3 == 1 else 150
        activation_func = 'tanh' if i % 3 == 1 else ('sigmoid' if i % 3 == 2 else 'relu')
        optimizer = 'sgd'
    elif i == 13:
        num_hidden_layers = 2
        num_neurons = 100
        activation_func = 'tanh'
        optimizer = 'adam'
    elif i == 14:
        num_hidden_layers = 2
        num_neurons = 150
        activation_func = 'tanh'
        optimizer = 'adam'
    else:
        num_hidden_layers = 3
        num_neurons = 150
        activation_func = 'tanh'
        optimizer = 'adam'
    
    model = build_model(num_hidden_layers, num_neurons, activation_func, optimizer)
    
    history = model.fit(X_train, y_train, epochs=num_epochs, validation_split=0.1)
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"Model {i}: Test accuracy = {test_acc:.4f}")
