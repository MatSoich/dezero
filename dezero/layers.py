from dezero.core import Parameter
import weakref
import numpy as np
import dezero.functions as F


class Layer:
    def __init__(self):
        #　_paramsというインスタンス変数の定義
        self._params = set()
    # Layerクラス内でインスタンス変数を設定する際に呼び出される特殊メソッド
    # インスタンス変数の名前がname, インスタンス変数の値がvalueとして引数に渡される。
    def __setattr__(self, name, value):
        if isinstance(value, (Parameter, Layer)): #①Layerも追加する。
            self._params.add(name)
        super().__setattr__(name, value)
    
    def __call__(self, *inputs):
        # forwardメソッドを呼び出す。
        outputs = self.forward(*inputs)
        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        # 入力と出力を弱参照で保持。
        self.inputs = [weakref.ref(x) for x in inputs]
        self.outputs = [weakref.ref(y) for y in outputs]
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, inputs):
        raise NotImplementedError()
    # Layerインスタンスが持つParameterインスタンスを取り出す。
    def params(self):
        # for + yieldで処理を逐次実行できる。
        for name in self._params:
            obj = self.__dict__[name]
            if isinstance(obj, Layer):
                # yield from によって、別のジェネレータから新しいジェネレータを読んでいる。
                yield from obj.params() #②Layerの場合、再起的にパラメータを取り出す。
            else:
                yield obj
    # 全てのParameterの勾配をリセット
    def cleargrads(self):
        for param in self.params():
            param.cleargrad()

class Linear(Layer):
    def __init__(self, out_size, nobias = False, dtype = np.float32, in_size=None):
        super().__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dtype = dtype

        self.W = Parameter(None, name = 'W')
        if self.in_size is not None:
            self._init_W()
        
        if nobias:
            self.b = None
        else:
            self.b = Parameter(np.zeros(out_size, dtype=dtype), name='b')
    
    def _init_W(self):
        I, O = self.in_size, self.out_size
        W_data = np.random.randn(I,O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = W_data

    def forward(self, x):
        # データを流すタイミングで初期化。
        if self.W.data is None:
            self.in_size = x.shape[1]
            self._init_W()

        y = F.linear(x, self.W, self.b)
        return y

