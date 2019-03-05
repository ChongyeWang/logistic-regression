import h5py
import numpy as np
from random import randint

"""
Load the mnist dataset.
"""
def load_data(filename):
    data = h5py.File(filename, 'r')
    x_train = np.float32(data['x_train'][:])
    y_train = np.int32(np.array(data['y_train'][:, 0]))
    x_test = np.float32(data['x_test'][:])
    y_test = np.int32(np.array(data['y_test'][:, 0]))
    data.close()
    return x_train, y_train, x_test, y_test

"""
Softmax function for multiclass logistic regression.
"""
def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))

"""
Forward function for calculation process.
"""
def forward(x, weight):
    z = np.dot(weight, x)
    return softmax(z)

"""
Update the gradient.
"""
def update_gradient(x, y, prob, gradient, num_classes):
    for i in range(num_classes):
        gradient[i] = prob[i] * x
    gradient[y] -= x
    return gradient

def main():
    lr = 0.0001 # learning rate
    epochs = 30 # number of epochs
    num_features = 784 # number of features
    num_classes = 10 # number of output classes
    weight = np.random.randn(num_classes, num_features) * 0.001 # initial weight
    gradient = weight[:] # inital gradient
    
    x_train, y_train, x_test, y_test = load_data('MNISTdata.hdf5') # load mnist data
    for epoch in range(epochs):
        num_of_correct = 0 # count the number of correct estimated result.
        for _ in range(len(x_train)):
            random_idx = randint(0, len(x_train) - 1)
            selected_x = x_train[random_idx]
            selected_y = y_train[random_idx]
            prob = forward(selected_x, weight)
            estimated_result = np.argmax(prob)
            num_of_correct += estimated_result == selected_y
            gradient = update_gradient(selected_x, selected_y, prob, gradient, num_classes)
            weight = weight - lr * gradient
        print(float(num_of_correct) / float(len(x_train)))

    def test():
        num_correct = 0
        for i in range(len(x_test)):
            x = x_test[i]
            y = y_test[i]
            prob = forward(x, weight)
            estimated_result = np.argmax(prob)
            num_correct += estimated_result == y
        print("Test Accuracy : ")
        print(float(num_correct) / float(len(x_test)))
    
    # Test 
    test()
    #Test Accuracy : 0.9132

if __name__ == "__main__":
    main()

