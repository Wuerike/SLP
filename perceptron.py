import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, max_training_epoch = 100, learning_rate = 0.01):
        self.max_training_epoch = max_training_epoch
        self.learning_rate = learning_rate
        self.convergence_counter = 0
        self.convergence_factor = 3
        self.weights = None
        self.bias = 0
        self.previous_bias = 0

    # Method that plots a data set and the decision boundary line
    def plot_data(self, values, labels, plot_name):
            # Plot the received values / labels
            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.set_title(plot_name)
            plt.scatter(values[:,0], values[:,1] ,marker='x', c=labels)

            # Get X min and max values
            x1 = np.amin(values[:,0])
            x2 = np.amax(values[:,0])

            # From the perceptron predict function w1*x1 + w2*x2 + b = 0
            # Being x1 = x and x2 = y ---> y = (-w1*x - b)/w2
            # Get Y values to max and min X values
            y1 = (-self.weights[0] * x1 - self.bias) / self.weights[1]
            y2 = (-self.weights[0] * x2 - self.bias) / self.weights[1]

            # Plot the decision boundery
            ax.plot([x1, x2], [y1, y2], 'k')

            # show the plot in a new window
            plt.show()

    # Method that checks if the training has converged
    def check_convergence(self):
        if self.previous_bias == self.bias:
            # If repeating bias, increment the counter
            self.convergence_counter += 1
        else:
            # Otherwise, save the bias and set the counter to 0
            self.previous_bias = self.bias
            self.convergence_counter = 0

        # Returns true when the counter is equal to convergence_factor
        if self.convergence_counter == self.convergence_factor:
            return True
        else:
            return False

    # Method that returns the accuracy for given predited labels and real labels values
    def accuracy(self, real_labels, predited_labels):
        accuracy = np.sum(real_labels == predited_labels) / len(real_labels)
        return accuracy

    # Method to predict labels based on its inputs
    def predict(self, inputs):
        # bias value + scalar product between inputs and weights
        dot_product = np.dot(inputs, self.weights) + self.bias

        # apply the activation function, returning 0 or 1        
        return np.where(dot_product>=0, 1, 0)

    # Method to train an one layer perceptron neural artificial network
    def train(self, training_inputs, training_labels):
        # Gets the training data dimensions (2 atributtes make the plot visible)
        n_samples, n_atributtes = training_inputs.shape
        self.weights = np.zeros(n_atributtes)
        
        for epoch in range(self.max_training_epoch):
            # Apply the traning rule to ajust weights and bias
            for inputs, label in zip(training_inputs, training_labels):
                prediction = self.predict(inputs)
                self.weights += self.learning_rate * (label - prediction) * inputs
                self.bias += self.learning_rate * (label - prediction)

            # Early stops the training when repeating the same bias for convergence_factor times
            if self.check_convergence():
                self.plot_data(training_inputs, training_labels, "Training complete with " + str(epoch + 1) + " epochs")
                break

            # plot each traning epoch, plot window should be closed to see next interation
            self.plot_data(training_inputs, training_labels, "Training epoch: " + str(epoch + 1))

def test():
    # Create a data set as described in the parameters
    values, labels = datasets.make_blobs(n_samples=300, n_features=2, centers=2, cluster_std=1.25, random_state=6)
    # Split the data set in train data set an test data set
    train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.2, random_state=101)

    p = Perceptron(max_training_epoch=100, learning_rate=0.001)
    p.train(train_values, train_labels)
    predictions = p.predict(test_values)

    print("Test data-set accuracy", p.accuracy(test_labels, predictions))

    p.plot_data(test_values, test_labels, "PERCEPTRON: TEST DATA-SET")

if __name__ == '__main__':
    test()