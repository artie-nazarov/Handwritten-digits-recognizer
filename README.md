# Build a Neural Network from scratch analytically - Handwritten digits

### Overview
This is a Neural Network built from scratch using the NumPy library to solve the MNIST handwritten digit classifier. The architecture of the neural net was designed by diagnosing a bias vs. variance problem using a leraning curve and a regularization parameter analysis.
Results: Test set accuracy: **96.3%**

### Goal

The goal of this project is to build a neural network from scratch and analytically asses it's performance using varying models in order to find the best NN architecture for a given dataset. 

### Dataset

This project uses the MNIST handwritten digits dataset. The dataset used in this project can be found at: https://www.kaggle.com/c/digit-recognizer/data


## Building the Neural Network

### Tools

* [Numpy](https://numpy.org/doc/stable/) - Package for scientific computing

### Activation function

This neural network is using a **tanh** activation function on the hidden layers and a **softmax** activation function on the output layer


### Initial NN architecture

Initially, I have chosen an arbitrary neural network architecture in order to test the first "quick and dirty" implementation of the cost function, gradient, and learning curve.

Initial architecture consists of 3 layers (1 hidden layer). The input layer consists of 784 neurons (28x28 pixels), 400 neurons in the hidden layer (rougly a half of the input size), and 10 output nodes (classifies digits 0-9)

### Results

After analytically training the Network and assessing the learning curve behavior, the best Neural Network architecture was the following...

**Final NN Architecture:**  
Input layer: 784 units (28*28 pxls)  
1 Hidden Layer: 85 units <--- Main adjustment  
Output layer: 10 units (10 digits 0-9)
Learning rate: 0.3

Test set accuracy after 3000 iteretions: **96.3%**  

