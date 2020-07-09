import numpy as np
import matplotlib.pyplot as plt

def import_data():
    X = np.genfromtxt("train_X_lr.csv", delimiter=',', dtype=np.float64, skip_header=1)
    Y = np.genfromtxt("train_Y_lr.csv", delimiter=',', dtype=np.float64)
    return X,Y

def J(X, Y, W): # Returns the value of cost function(J)
    error = np.dot(X,W) - Y
    j = (1/(2*m)) * np.dot(error.T,error)
    return j.squeeze()

def dJ(X, Y, W): # Returns a vector of partial derivatives of J.
    error = np.dot(X,W) - Y
    dj = (1/m) * np.dot(X.T,error)
    return dj

def gradient_descent_W(X, Y, iterations, learning_rate): #Returns the vector of parameters W.
    global instances_of_costfn
    
    W = np.array([[-3850.738], [ 1.98], [  68.354], [  42.932], [   29.608]])
    for i in range(iterations):
        
        instances_of_costfn = np.append(instances_of_costfn, J(X, Y, W))#Appending new costfn
        
        W = W - learning_rate*dJ(X, Y, W)
    return W

if __name__ == "__main__":

    global m,n

    train_X, train_Y = import_data()
    m = len(train_Y)
    n = len(train_X[0])
    train_X = np.insert(train_X, 0, 1, axis=1)
    train_Y = train_Y.reshape((m,1)) # X and Y are perfect upto here

    instances_of_costfn = np.array([])
    iterations, alpha = 100000, 0.00021
    W = gradient_descent_W(train_X, train_Y, iterations, alpha)

    x = np.linspace(0, iterations-1, num=iterations)
    plt.plot(x, instances_of_costfn) # remove the third argument for a line graph without o dots.
    plt.grid() #fills background with grids
    plt.axvline() #adds x=0 (vertical axis) to the graph
    plt.axhline() #adds y=0 (horizontal axis) to the graph
    plt.show()

    mse = 2 * J(train_X, train_Y, W)

    print(W, instances_of_costfn[-1])
    print(mse)