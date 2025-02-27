import Layers
import Loss

class MLP:
    def __init__(self, layers: list[Layers], loss_fn:Loss, lr:float)->None:
        """
        Multi-Layer Perceptron (MLP) class.
        Arguments:
        - layers: List of layers (e.g., Linear, ReLU, etc.).
        - loss_fn: Loss function object (e.g., CrossEntropy, MSE).
        - lr: Learning rate.
        """
        self.layers = layers
        self.loss_fn = loss_fn
        self.lr  = lr

    def __call__(self, inp:np.ndarray)->np.ndarray:
        """Makes the model callable"""
        return self.forward(inp)

    def forward(self, inp:np.ndarray)->np.ndarray:
        """Passing the input array through each layer sequentially"""
        for layer in self.layers:
            inp = layer.forward(inp)
        return inp

    def loss(self, prediction:np.ndarray, target:np.ndarray) -> float:
        return self.loss_fn(prediction, target)

    def backward(self)->None:
        """Perform backpropogation by propagating the gradient backwards through the gradients and learning rate"""
        up_grad = self.loss_fn.backward()
        for layer in reversed(self.layers):
            up_grad = layer.backward(up_grad)

    def update(self) -> None:
        for layer in self.layers:
            layer.step(self.lr)
    
    def train(self, x_train: np.ndarray, y_train: np.ndarray, epochs: int, batch_size: int) -> np.ndarray:
        losses = np.empty(epochs)
        for epoch in (pbar := trange(epochs)):
            running_loss = 0.0
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]

                #Forward Pass
                prediction = self.forward(x_batch)

                #Compute loss
                running_loss += self.loss(prediction, y_batch) * batch_size

                #Backward Pass
                self.backward()

                #Update parameters
                self.update()

            running_loss /= len(x_train)
            pbar.set_description(f"Loss:{running_loss:.3f}")
            losses[epoch] = running_loss

        return losses


                