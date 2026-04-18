import math

class Value:
    def __init__(self, value, _parents=(), _op=''):
        self.value=value
        self.grad=0
        self._backward=lambda: None
        self._prev=set(_parents)
    
    def __repr__(self):
        return f"V:{self.value} G:{self.grad}"
    
    def __add__(self,other):
        out = Value(self.value+other.value,(self,other),'+')
        def _backward():
            self.grad+=out.grad * 1
            other.grad+=out.grad * 1
        out._backward=_backward
        return out

    def __sub__(self,other):
        out = Value(self.value-other.value,(self,other),'-')
        def _backward():
            self.grad+=out.grad * 1
            other.grad+=out.grad * (-1)
        out._backward=_backward
        return out
    
    def __mul__(self,other):
        out = Value(self.value*other.value,(self,other),'*')
        def _backward():
            self.grad+=other.value*out.grad
            other.grad+=self.value*out.grad
        out._backward=_backward
        return out
    
    def __pow__(self,value:int | float):
        out = Value(self.value**value,(self,),f'**{value}')
        def _backward():
            self.grad+=value*(self.value**(value-1))*out.grad
        out._backward=_backward
        return out
    
    # def __truediv__(self,value:int | float):
    #     out = Value(self.value*(value**(-1)),(self,),f'/{value}')
    #     def _backward():
    #         self.grad+=value**(-1)*out.grad
    #     out._backward=_backward
    #     return out
    def __truediv__(self,other):
        out = Value(self.value*(other.value**(-1)),(self,other),f'/{other.value}')
        def _backward():
            self.grad+=other.value**(-1)*out.grad
            other.grad+=(-1)*(self.value*(other.value**(-2)))*out.grad
        out._backward=_backward
        return out

    def __rtruediv__(self,other):
        out = Value(other.value*(self.value**(-1)),(self,other),f'{other.value}/{self.value}')
        def _backward():
            self.grad+=(-1)*(other.value*(self.value**(-2)))*out.grad
            other.grad+=(self.value**(-1))*out.grad
        out._backward=_backward
        return out

    def relu(self):
        out = Value(0 if self.value<0 else self.value,(self,),'ReLU')
        def _backward():
            self.grad+=(out.value>0)*out.grad
        out._backward=_backward
        return out

    def topo_sort(self):
        topo=[]
        visited=set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        return topo

    def backward(self):
        topo = self.topo_sort()
        self.grad=1
        for node in reversed(topo):
            node._backward()

    def zero_grads(self):
        topo = self.topo_sort()
        for node in topo:
            node.grad=0

    def __rpow__(self,value:int | float):
        out = Value(value**self.value,(self,),f'exp({self.value})')
        def _backward():
            self.grad+=value**self.value*out.grad*math.log(value)
        out._backward=_backward
        return out

    def log(self):
        out = Value(math.log(self.value),(self,),'log')
        def _backward():
            self.grad+=(1/self.value)*out.grad
        out._backward=_backward
        return out

x1=Value(2.0)
x2=Value(0.0)
w11=Value(-3.0)
w12=Value(1.0)
w13=Value(2.0)
w21=Value(1.0)
w22=Value(-2.0)
w23=Value(-3.0)
b1=x1*w11+x2*w21
b2=x1*w12+x2*w22
b3=x1*w13+x2*w23
b1=b1.relu()
b2=b2.relu()
b3=b3.relu()
w1=Value(-3.0)
w2=Value(1.0)
w3=Value(-1.0)
y=b1*w1+b2*w2+b3*w3
y.backward()
print(b1) # V:0 G:-3
print(b2) # V:2 G:1
print(b3) # V:4 G:-1
print(x1) # V:2 G:-1
