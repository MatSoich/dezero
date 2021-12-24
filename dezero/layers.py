import os
import weakref
import numpy as np
from dezero import cuda
from dezero.core import Parameter
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

    def to_cpu(self):
        for param in self.params():
            param.to_cpu()

    def to_gpu(self):
        for param in self.params():
            param.to_gpu()

    #Parameterを入れ子になっていないディクショナリとして取り出す関数
    def _flatten_params(self, params_dict, parent_key=""):
        for name in self._params:
            obj = self.__dict__[name]
            key = parent_key + '/' + name if parent_key else name

            if isinstance(obj, Layer):
                obj._flatten_params(params_dict, key)
            else:
                params_dict[key] = obj

    def save_weights(self, path):
        self.to_cpu()

        params_dict = {}
        self._flatten_params(params_dict)
        array_dict = {key: param.data for key, param in params_dict.items() if param is not None}
        try:
            np.savez_compressed(path, **array_dict)
        except (Exception, KeyboardInterrupt) as e:
            if os.path.exists(path):
                os.remove(path)
            raise
    
    def load_weights(self, path):
        npz = np.load(path)
        params_dict = {}
        self._flatten_params(params_dict)
        for key, param in params_dict.items():
            param.data = npz[key]




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
    
    def _init_W(self, xp=np):
        I, O = self.in_size, self.out_size
        W_data = xp.random.randn(I,O).astype(self.dtype) * np.sqrt(1/I)
        self.W.data = W_data

    def forward(self, x):
        # データを流すタイミングで初期化。
        if self.W.data is None:
            self.in_size = x.shape[1]
            xp = cuda.get_array_module(x)
            self._init_W(xp)

        y = F.linear(x, self.W, self.b)
        return y

