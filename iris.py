import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from perceptron import Perceptron

# read the data set
data = pd.read_csv("iris.csv")

# Get data only from setosa and versicolor classes
# Get the atributtes sepal length and petal length
values = data.iloc[0:100, [0, 2]].values
labels = data.iloc[0:100, 4].values

# Redefine setosa class as 0 and versicolor as 1
labels = np.where(labels == 'Iris-setosa', 0, 1)

# Split the data set in train data set an test data set
train_values, test_values, train_labels, test_labels = train_test_split(values, labels, test_size=0.2, random_state=101, stratify=labels)

# Train the perceptron 
p = Perceptron(max_training_epoch=100, learning_rate=0.001)
p.train(train_values, train_labels)

# Prdict the test values data set
predictions = p.predict(test_values)

print("Test data-set accuracy", p.accuracy(test_labels, predictions))

# Shows a plot with the test data set
p.plot_data(test_values, test_labels, "PERCEPTRON: TEST DATA-SET")


# Plot the all the setosa and versicolor values
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.set_title("COMPLETE IRIS DATA-SET")

plt.scatter(values[:50, 0], values[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(values[50:100, 0], values[50:100, 1], color='blue', marker='x', label='versicolor')
plt.xlabel('petal length')
plt.ylabel('sepal length')
plt.legend(loc='upper left')

# Get X min and max values
x1 = np.amin(values[:,0])
x2 = np.amax(values[:,0])

# From the perceptron predict function w1*x1 + w2*x2 + b = 0
# Being x1 = x and x2 = y ---> y = (-w1*x - b)/w2
# Get Y values to max and min X values
y1 = (-p.weights[0] * x1 - p.bias) / p.weights[1]
y2 = (-p.weights[0] * x2 - p.bias) / p.weights[1]

# Plot the decision boundery
#plt.plot([x1, x2], [y1, y2], 'k')
plt.show()
