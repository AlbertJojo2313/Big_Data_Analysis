##Creating my own neural network
import numpy as np

##Creating class layer for the layers
class Layer:
    def __init__(self):
        self.inp = None
        self.out = None
    def __call__(self, inp:np.ndarray)->np.ndarray:
        return self.forward(inp)
    
    def forward(self,inp:np.ndarray)->np.ndarray:
        raise NotImplementedError
    

    def backward(self,up_grad:np.ndarray)->np.ndarray:
        raise NotImplementedError
    
    def step(self, lr:float) ->None:
        pass


class Linear(Layer):
    super.__init__()
    self.w = 0.1 * np.random.randn(in_dim,out_dim)
    self.b = np.zeros((1,out_dim))
    self.dw = np.zeros_like(self.w)
    self.db = np.zeros_like(self.b)

    def forward(self, inp:np.ndarray)-> np.ndarray:
        self.inp = inp
        self.out = np.dot(inp, self.w) + self.b
        return self.out
    
    def backward(self, up_grad:np.ndarray)-> np.ndarray:
        """Backpropograte the gradients through this layer"""
        self.dw = np.dot(self.inp.T, up_grad) #Gradient wrt weights
        self.db = np.sum(up_grad,axis=0, keepdims=True) #Gradient wrt biases
        down_grad = np.dot(up_grad,self.w.T)
        return down_grad
    
    def step(self, lr:float)->None:
        """Update the weights and biases using the gradients"""
        self.w -= lr * self.dw
        self.b -= lr * self.db
    
###Activation Functions
class Sigmoid(Layer):
    def forward(self, inp:np.ndarray) ->np.ndarray:
        """Sigmoid Activation: f(x) = 1/(1+exp(-x))"""
        self.out = 1/(1+np.exp(-inp))
        return self.out
    
    def backward(self, up_grad:np.ndarray)->np.ndarray:
        down_grad = self.out * (1-self.out) * up_grad
        return down_grad
    

class ReLU(Layer):
    def forward(self, inp: np.ndarray)->np.ndarray:
        self.inp = inp
        self.out = np.maximum(0, inp)
        return self.out
    
    def backward(self, up_grad:np.ndarray)->np.ndarray:
        down_grad = up_grad * (self.inp > 0)
        return down_grad

class Softmax(Layer):
    def forward(self, inp:np.ndarray)->np.ndarray:
        """f(x) = exp(x) / sum(exp(x))"""
        # Subtract max to prevent overflow
        exp_values = np.exp(inp - np.max(inp, axis=1, keepdims=True))
        self.out = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return self.out

    def backward(self, up_grad:np.ndarray)->np.ndarray:
        """Backward passs for softmax using Jacobian Matrix"""
        down_grad = np.empty_like(up_grad)
        for i in range(up_grad.shape[0]):
            single_output = self.out[i].reshape(-1,1)
            jacobian = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            down_grad[i] = np.dot(jacobian, up_grad[i])
        return down_grad