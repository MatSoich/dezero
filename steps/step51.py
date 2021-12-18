if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import dezero
from dezero import DataLoader, optimizers
from dezero.datasets import Spiral
from dezero.models import MLP
import dezero.functions as F

max_epoch = 5
batch_size = 100
hidden_size = 1000


train_set = dezero.datasets.MNIST(train=True)
test_set = dezero.datasets.MNIST(train=False)
train_loader = DataLoader(train_set, batch_size)
test_loader = DataLoader(test_set, batch_size, shuffle=False)

# model = MLP((hidden_size, 10))
model = MLP((hidden_size, hidden_size, 10), activation = F.relu)
# optimizer = optimizers.SGD().setup(model)
optimizer = optimizers.Adam().setup(model)


for epoch in range(max_epoch):
    sum_loss, sum_acc = 0, 0
    for x, t in train_loader: # ①訓練用のミニバッチデータ
        y = model(x)
        loss = F.softmax_cross_entropy(y,t)
        acc = F.accuracy(y, t) # ②訓練データの認識精度
        model.cleargrads()
        loss.backward()
        optimizer.update()
        
        sum_loss += float(loss.data) * len(t)
        sum_acc += float(acc.data) * len(t)
    print('epoch: {}'.format(epoch+1))
    print('train loss: {: .4f}, accuracy: {: .4f}'.format(sum_loss / len(train_set), sum_acc / len(train_set)))

    sum_loss, sum_acc = 0, 0
    with dezero.no_grad(): #③勾配不要モード
        for x, t in test_loader: #④訓練用のミニバッチデータ 
            y = model(x)
            loss = F.softmax_cross_entropy(y,t)
            acc = F.accuracy(y,t) #⑤テストデータの認識精度
            sum_loss += float(loss.data) * len(t)
            sum_acc += float(acc.data) * len(t)

    print('test loss: {: .4f}, accuracy: {: .4f}'.format(sum_loss / len(test_set), sum_acc / len(test_set)))