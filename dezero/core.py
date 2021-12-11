if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import contextlib
import numpy as np
import weakref
import dezero

class Variable:
    #ndarrayが前に来たときの演算子を用いた計算でndarrayより優先してメソッドを読んでもらうための設定
    __array_priority__ = 200
    # 変数に名前をつけれるように改良
    def __init__(self, data, name=None):
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError('{} is not supported'.format(type(data)))
        
        self.data = data
        self.name = name
        self.grad = None
        # 変数を作成した関数を記憶するための変数
        self.creator = None
        # 世代の初期値
        self.generation = 0

    def set_creator(self,func):
        self.creator = func
        #出力は親となる関数より1世代大きくなる。
        self.generation = func.generation + 1
    
    #逆伝播の中での計算を逆伝播無効モードも踏まえて効率的に行えるように変更。（create_graph追加）
    def backward(self, retain_grad=False, create_graph=False):
        if self.grad is None:
            # self.grad = np.ones_like(self.data)
            # 逆伝播でもつながりを作るために、Variableインスタンスになるように修正
            self.grad = Variable(np.ones_like(self.data))

        funcs = []
        seen_set = set()

        # 関数の中でしか関数を使わない時
        # 親となる関数の中で指定している変数にアクセスする必要がある。（funcsやseen_set）
        #　上記2点を満たすため、add_func関数はbackward関数の中に記載される。
        def add_func(f):
            # seen_setを使って、backwardが2回呼ばれるのを防ぐ。
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                # ここでは全ての要素をソートしているが、より良い方法がある。例えば優先度つきキュー。Pythonではheadqとか。
                funcs.sort(key=lambda x: x.generation)
        
        add_func(self.creator)

        while funcs:
            f = funcs.pop() # 関数取得
            # 出力変数outputsの微分をリストにまとめる。弱参照処理に合わせて()を追記。
            gys = [output().grad for output in f.outputs]
            # while usingの中で逆伝播をするように変更（Step32）
            with using_config('enable_backprop', create_graph):
                # fの逆伝播を呼び出す。
                gxs = f.backward(*gys)
                # gxsがタプルでない時タプルへ変換。
                if not isinstance(gxs, tuple):
                    gxs = (gxs,)
                
                for x, gx in zip(f.inputs, gxs):
                    if x.grad is None:
                        x.grad = gx
                    else:
                        x.grad = x.grad + gx
                        
                    if x.creator is not None:
                        add_func(x.creator) #１つ前の関数をリストに追加。

        if not retain_grad:
            for y in f.outputs:
                y().grad = None #yはweakrefなので()がいる。
 
    def cleargrad(self):
        self.grad = None
    
    #Variableクラス内のメソッドとして使用するためにfunctionsの他にVariableクラスにも追記。
    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return dezero.functions.reshape(self, shape)

    #Variableクラス内のメソッドとして使用するためにfunctionsの他にVariableクラスにも追記。
    def transpose(self):
        return dezero.functions.transpose(self)

    #Variableクラス内のメソッドとして使用するためにfunctionsの他にVariableクラスにも追記。 
    def sum(self, axis=None, keepdims=False):
        return dezero.functions.sum(self, axis, keepdims)

    # Variableインスタンスをndarrayのインスタンスのように見せるための実装。(必要に応じて幾つでも設定可能。)
    
    # インスタンス変数としてアクセスするためのデコレータ
    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size
    
    @property
    def dtype(self):
        return self.data.dtype

    @property
    def T(self):
        return dezero.functions.transpose(self)


    # PYTHOのlen関数が使えるように拡張
    # __len__という特殊メソッドを実装。
    def __len__(self):
        return len(self.data)

    # print関数が使えるように拡張
    # 必要なら文字列の中身を変更可能。
    def __repr__(self):
        if self.data is None:
            return 'variable(None)'
        p = str(self.data).replace('\n', '\n' + ' ' * 9)
        return 'variable(' + p + ')'
    
    #def __mul__(self, other):
    #    return mul(self, other)

class Parameter(Variable):
    pass
 
#引数と戻り値をリストに変更                
class Function:
    def __call__(self, *inputs):
        # inputsの各要素をVariableインスタンスに変換（Variableインスタンスではない場合）
        inputs = [as_variable(x) for x in inputs]
        #　リスト内包表記でinputsの各要素xのリストxsを作成。
        xs = [x.data for x in inputs]
        ys = self.forward(*xs) #アスタリスクをつけて渡すことでリストをアンパックして関数に渡す
        if not isinstance(ys, tuple):
            ys = (ys,)
        # outputも同様にリスト内包表記で実装
        outputs = [Variable(as_array(y)) for y in ys] 
        # 逆伝播有効モード実行時のみ実施。
        if Config.enable_backprop:
            #関数の世代は入力の最大と等しくなる。
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self) #出力変数に生みの親を覚えさせる
        self.inputs = inputs #入力された変数を覚えさせる。
        self.outputs = [weakref.ref(output) for output in outputs] #出力も覚える。弱参照で。
        return outputs if len(outputs) > 1 else outputs[0]
        
    def forward(self,x):
        raise NotImplementedError()
    
    def backward(self,gy):
        raise NotImplementedError()

class Config:
    enable_backprop = True


class Add(Function):
    def forward(self,x0, x1):
        #ブロードキャスト機能のための対応
        self.x0_shape, self.x1_shape = x0.shape, x1.shape

        y = x0 + x1
        return y
    def backward(self, gy):
        gx0, gx1 = gy, gy
        if self.x0_shape != self.x1_shape: # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Mul(Function):
    def forward(self, x0, x1):
        y = x0 * x1
        return y
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        # Variableインスタンスに変更したことへの対応
        x0, x1 = self.inputs
        gx0 = gy * x1
        gx1 = gy * x0
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Neg(Function):
    def forward(self, x):
        return -x
    def backward(self, gy):
        return -gy

class Sub(Function):
    def forward(self, x0, x1):
        self.x0_shape, self.x1_shape = x0.shape, x1.shape # for broadcast
        y = x0 - x1
        return y
    def backward(self, gy):
        gx0 = gy
        gx1 = -gy
        if self.x0_shape != self.x1_shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, self.x0_shape)
            gx1 = dezero.functions.sum_to(gx1, self.x1_shape)
        return gx0, gx1

class Div(Function):
    def forward(self, x0, x1):
        y = x0 / x1
        return y
    def backward(self, gy):
        # x0, x1 = self.inputs[0].data, self.inputs[1].data
        # Variableインスタンスに変更したことへの対応
        x0, x1 = self.inputs
        gx0 = gy /x1
        gx1 = gy * (- x0 / x1 ** 2 )
        if x0.shape != x1.shape:  # for broadcast
            gx0 = dezero.functions.sum_to(gx0, x0.shape)
            gx1 = dezero.functions.sum_to(gx1, x1.shape)
        return gx0, gx1

class Pow(Function):
    def __init__(self, c):
        self.c = c

    def forward(self,x):
        y = x ** self.c
        return y
    
    def backward(self, gy):
        # x = self.inputs[0].data
        # Variableインスタンスに変更したことへの対応
        x, = self.inputs
        c = self.c
        gx = c * x ** (c - 1) * gy
        return gx



def as_array(x):

    if np.isscalar(x):
        return np.array(x)
    return x

# ndarrayインスタンスをVaraibleインスタンスに変換
def as_variable(obj):
    if isinstance(obj, Variable):
        return obj
    return Variable(obj)

#using_Config関数をwith構文で使うためのデコレータ
@contextlib.contextmanager
# nameにはConfigの属性名（class属性名を指定。）
def using_config(name, value):
    # 前処理（古い値を取り出して新しいいvalueを設定。）
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    # 本処理(with構文の中の処理)
    try:
        yield 
    #後処理
    finally:
        #valueを元に戻す
        setattr(Config, name, old_value)

# with構文の後にusing...と何回も書くのは面倒なので
def no_grad():
    return using_config('enable_backprop', False)

def add(x0, x1):
    x1 = as_array(x1)
    return Add()(x0, x1)

def mul(x0, x1):
    x1 = as_array(x1)
    return Mul()(x0, x1) 

def neg(x):
    return Neg()(x)

def sub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x0, x1)

def rsub(x0, x1):
    x1 = as_array(x1)
    return Sub()(x1, x0)

def div(x0, x1):
    x1 = as_array(x1)
    return Div()(x0, x1)

def rdiv(x0, x1):
    x1 = as_array(x1)
    return Div()(x1, x0)

def pow(x,c):
    return Pow(c)(x)

# Variableに対して演算子の設定を行うための関数
def setup_variable():
    Variable.__mul__ = mul
    Variable.__rmul__ = mul
    Variable.__add__ = add
    Variable.__radd__ = add
    Variable.__neg__ = neg
    Variable.__sub__ = sub
    Variable.__rsub__ = rsub
    Variable.__truediv__ = div
    Variable.__rtruediv__ = rdiv
    Variable.__pow__ = pow