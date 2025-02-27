import numpy as np

class Loss:
    def __init__(self):
        self.prediction = None
        self.target = None
        self.loss = None
    
    def __call__(self, prediction: np.ndarray, target: np.ndarray)->float:
        return self.forward(prediction=prediction, target=target)

    
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> float:
        raise NotImplementedError

    def backward(self) -> np.ndarray:
        raise NotImplementedError


class CrossEntropy(Loss):
    def forward(self, prediction:np.ndarray, target:np.ndarray)-> float:
        self.prediction = prediction
        self.target = target

        clipped_pred = np.clip(prediction, 1e-12,1.0)

        self.loss = -np.mean(np.sum(target * np.log(clipped_pred), axis=1))

        return self.loss
    
    def backward(self)->np.ndarray:
        """Gradient of Cross Entropy Loss"""
        grad = -self.target / self.prediction / self.target.shape[0]
        return grad

class MSE(Loss):
    def forward(self, prediction:np.ndarray, target:np.ndarray)->float:
        self.prediction = prediction
        self.target = target

        self.loss = np.mean((prediction - target) ** 2)
        return self.loss

    def backward(self)->np.ndarray:
        grad = 2 * (self.prediction - self.target) / self.target.size

        return grad

