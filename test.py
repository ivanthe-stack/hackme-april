import numpy as np
import argparse
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', required=True, help='model file to load')
parser.add_argument('-s', '--seed', type=int, default=42, help='random seed')
args = parser.parse_args()

class MLP:
    def __init__(self, layers):
        self.layers = layers
        self.weights = []
        self.biases = []

    def forward(self, x):
        for i in range(len(self.weights)):
            x = x @ self.weights[i] + self.biases[i]
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)
        return np.exp(x - x.max(axis=1, keepdims=True))

    def load(self, path):
        data = np.load(path, allow_pickle=True)
        self.weights = list(data['weights'])
        self.biases = list(data['biases'])
        self.layers = list(data['layers'])

np.random.seed(args.seed)

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state=args.seed)
X_train = X_train / 16.0
X_test = X_test / 16.0

mlp = MLP([0])
mlp.load(args.model)
print(f"Loaded model from {args.model}")
print(f"Architecture: {mlp.layers}")

pred = np.argmax(mlp.forward(X_test), axis=1)
acc = np.mean(pred == y_test)
test_out = mlp.forward(X_test)
test_loss = -np.mean(np.log(test_out[np.arange(len(y_test)), y_test] + 1e-8))

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {acc:.4f}")