import numpy as np
from dezero import Variable
from dezero.utils import get_dot_graph

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
#計算処理
y = x0 + x1

#変数の命名
x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

txt = get_dot_graph(y, verbose=False)
print(txt)

# dotファイルとして保存
with open('sample.dot','w') as o:
    o.write(txt)