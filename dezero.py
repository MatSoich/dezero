# ゼロから作るDeep Learningのコードを書いてみるところ
class Variable:
    def __init__(self, data):
        self.data=data

import numpy as np
data = np.array(1.0)
x = Variable(data)
print(x.data)

# xに新しいデータを代入
x.data = np.array(2.0)
print(x.data)

"""
class Function:
    def __call__(self, input):
        x=input.data #データの取り出し処理
        y=x**2
        output=Variable(y) #Variableとして返す
        return output

x = Variable(np.array(10))
f = Function()
y = f(x)
print(type(y))
print(y.data)
"""

# Functionクラスは今後様々な関数を追加することになるので、Functionクラスを基底クラスとして全ての関数に関する共通機能を持つように実装する。
# 具体的な関数はFunctionクラスを継承したクラスで実装する。
class Function:
    def __call__(self, input):
        x=input.data
        y=self.forward(x)
        output=Variable(y) 
        return output

    def forward(self,x):
        raise NotImplementedError()

# 次にForwardクラスを継承して、入力された値を２乗するクラスを実装してみる。
class Square(Function):
    def forward(self,x):
        return x ** 2

x = Variable(np.array(10))
f = Square()
y = f(x)
print(type(y))
print(y.data)

class Exp(Function):
    def forward(self,x):
        return np.exp(x)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)
print(y.data)


def numerical_diff(f,x,eps=le-4):
    x0=
    x1=

