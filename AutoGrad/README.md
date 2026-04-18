# AutoGrad + Neural Network + Optimizer

Tiny from-scratch autograd + neural net playground in pure Python.

## [Quickstart (Karpathy-style scalar graph) - click for online demo](https://colab.research.google.com/drive/15uDB3PAIiDDnx1gcMbRDIn4lwCGprbW8?usp=sharing)

```python
from AutoGrad import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)

d = a * b
e = d + c
L = e.relu()

L.backward()

print("L:", L.value)
print("a.grad:", a.grad)
print("b.grad:", b.grad)
print("c.grad:", c.grad)
```

This is the same learning flow as micrograd examples:
- build a tiny computation graph by hand
- call `backward()` once on the final node
- inspect gradients on leaf nodes

## Train the tiny network

```bash
python3 optimizer.py
```

That runs one small demo loop with:
- `Network` from `NeuralNetwork.py`
- `GD_Optimizer` from `optimizer.py`
- cross-entropy over integer labels
