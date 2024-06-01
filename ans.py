import numpy
import numpy as np
import matplotlib.pyplot as plt

# Sigmoid activation function


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of sigmoid function
def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(1)
    W1 = np.random.randn(n_h, n_x)*np.sqrt(6/(n_h+n_x))
    b1 = np.random.randn(n_h, 1)*np.sqrt(6/(n_h+n_x))
    W2 = np.random.randn(n_y, n_h)*np.sqrt(6/n_h)
    b2 = 0

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters

def linear_activation_forward(T, W, b):
    Z= np.dot(W, T) + b
    A = sigmoid(Z)
    return A

def phi(x,g0):
    return (np.exp(-x/5)*np.sin(x)+g0).T
def NN(x,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]


    A1 = linear_activation_forward(x, W1, b1)
    A2 = np.sum(A1*W2.T,axis=0).reshape(-1,1)

    return A2

def NN_diff(W1,W2,A1):
    W1_0 = W1[:, 0]
    W1_0 = W1_0.reshape(-1, 1)
    C = A1 * (1 - A1)
    Y = C * W1_0 * W2.T
    Y = np.sum(Y, axis=0)
    Y = Y.reshape(-1, 1)
    return Y

def g_trial(g0,x,parameters):
    return g0+x.T*NN(x,parameters)
def func(x,parameters):
    return ((np.exp(-x/5))*np.cos(x))-g_trial(0,x,parameters)/5
def g_trial_diff(x,parameters,A1):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    return NN(x,parameters)+x.T*NN_diff(W1,W2,A1)

def cost_function(parameters,X,A1):
    dN_dx = g_trial_diff(X, parameters, A1)
    f = func(X,parameters)
    error_sq = (dN_dx - f) ** 2
    error_sq=np.sum(error_sq)
    return error_sq/len(X)
def gradients(parameters,X,A1):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    C = np.sum((A1 * (1 - A1) * (1 - 2*A1) * W2.T * W1),axis=1).reshape(-1,1)
    dW1=np.sum(np.dot(C,X),axis=1).reshape(-1,1)+np.sum(A1 * (1 - A1)*W2.T,axis=1).reshape(-1,1)
    dW1 = dW1.reshape(-1,1)
    dW2 = (np.sum(A1 * (1 - A1) * W1,axis=1)).T
    db1 = np.sum(A1 * (1 - A1) * (1 - 2*A1) * W1 * W2.T,axis=1).reshape(-1, 1)
    dW1 /= len(X)
    dW2 /= len(X)
    db1 /= len(X)
    return dW1,dW2,db1
def solve_ode(X,num_iters=1000,lmb=0.0007,lmb2=0.00005,lmb3=0.0005):
    (n_x, n_h, n_y) = 1, 20, 1
    parameters=initialize_parameters(n_x, n_h, n_y)
    # print(parameters)
    for i in range(num_iters):
        W1 = parameters["W1"]
        W2 = parameters["W2"]
        b1 = parameters["b1"]
        A1 = linear_activation_forward(X, W1, b1)
        dW1 ,dW2,db1= gradients(parameters,X,A1)
        cost=cost_function(parameters,X,A1)
        parameters["W1"]=W1-lmb*dW1
        parameters["W2"]=W2-lmb2*dW2
        parameters["b1"]=b1-lmb3*db1
        # print("here")
        if(i%10==0):
            print(cost)
    return parameters



# X=[[[0]],[[2]],[[3]],[[4]],[[5]],[[6]],[[7]],[[8]],[[9]],[[10]],[[11]],[[12]],[[13]],[[14]],[[15]],[[16]],[[17]],[[18]],[[19]]]
# X=np.array(X)
# X=[[[0.]],[[0.05]],[[0.1 ]],[[0.15]],[[0.2 ]],[[0.25]],[[0.3 ]],[[0.35]],[[0.4 ]],[[0.45]],[[0.5 ]],[[0.55]],[[0.6 ]],[[0.65]],[[0.7 ]],[[0.75]],[[0.8 ]],[[0.85]],[[0.9 ]],[[0.95]]]
# X=np.random.rand(20)
# X = X.reshape((1,20))
X=np.linspace(0,2,10)
# X=np.array([[0.2],[1.2],[2],[0.8],[1.8],[0.4],[1.4],[1.6],[1],[0.6]])
X=np.array(X)
X = X.reshape((1,10))
# indices = np.arange(len(X))
# np.random.shuffle(indices)
# X = X[indices]
print(X)
Y= X
p=solve_ode(X)

print(abs(g_trial(0,X,p)-phi(X,0)))
# print(g_trial(0,X,p),phi(X,0))

Y=Y.reshape((10,))
# # Y=[0,0.1,0.2,0.3,0.32,0.34,0.36,0.38,0.4,0.42,0.44,0.46,0.48,0.5,0.6,0.7,0.8,0.9,1]
O1=g_trial(0,X,p).reshape((10,))
O2=phi(X,0).reshape((10,))


plt.scatter(Y, O1, label='Line 1')
plt.scatter(Y, O2, label='Line 2')
plt.title("Two Line Plots on a Single Graph with Custom Styles")
plt.xlabel("X-axis Label")
plt.ylabel("Y-axis Label")
plt.legend()

plt.show()