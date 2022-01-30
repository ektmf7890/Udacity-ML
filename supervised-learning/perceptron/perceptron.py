import numpy as np
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    # Fill in code

    num_of_data = X.shape[0]
    for i in range(num_of_data):
        X_i = X[i, :]        
        y_i = y[i]
        class_idx = prediction(X_i, W, b)

        # misclassified to be above the line
        if class_idx == 1 and y_i == 0:
            W = W - (learn_rate * X_i.reshape((2, -1)))
            b = b - (learn_rate)
        # misclassified to be below the line
        elif class_idx == 0 and y_i == 1:

            W = W + (learn_rate * X_i.reshape(2, -1))
            b = b + (learn_rate)
        else: pass

    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 25):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max

    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    
    return boundary_lines

if __name__ == "__main__":
    data = np.loadtxt('data.csv', delimiter=',')
    X = data[:, :-1]
    y = data[:, -1]

    boundary_lines = trainPerceptronAlgorithm(X, y, 0.01, 10)
    # print(len(boundary_lines))
    # print(boundary_lines[1])

    import matplotlib.pyplot as plt
    
    plt.figure()
    axes = plt.gca()
    axes.set_ylim([-0.50, 1.50])
    
    x1_min = X[:, 0].min()
    x1_max = X[:, 0].max()

    for slope, intercept in boundary_lines:
        # print(slope[0], intercept[0])
        plt.plot([x1_min, x1_max], [x1_min * slope[0] + intercept[0], x1_max * slope[0] + intercept[0]])

    for i in range(X.shape[0]):
        x_1 = X[i, 0]
        x_2 = X[i, 1]
        y_i = y[i]

        if y_i == 0:
            color = "red"
        else:
            color = "blue"

        plt.scatter(x_1, x_2, c=color)
        
    
    plt.show()
