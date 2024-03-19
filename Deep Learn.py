"""线性模型"""
"""import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)*(y_pred - y)

w_list = []
mse_list = []

for w in np.arange(0.0,4.1,0.1):
    print("w = ",w)
    l_sum = 0
    for x_val,y_val in zip(x_data,y_data):
        y_pred_val = forward(x_val)
        l_sum += loss(x_val,y_val)
        print("\t",x_val,y_val,y_pred_val)
    w_list.append(w)
    mse_list.append(l_sum/3)
plt.plot(w_list, mse_list)
plt.show()
"""

"""梯度递减"""
"""
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x * w

def cost(xs,ys):
    cost = 0
    for x_val,y_val in zip(xs,ys):
        y_pred = forward(x_val)
        cost += (y_pred - y_val)**2
    return cost / len(xs)

def gra(xs,ys):
    gra = 0
    for x_val,y_val in zip(xs,ys):
        gra += 2*x_val*(x_val * w - y_val)
    return gra / len(xs)

print('Predict (before learning)',4,forward(4))
for epoch in range(100):
    cost_val = cost(x_data,y_data)
    gra_val = gra(x_data,y_data)
    w -= 0.01*gra_val
    print('Epoch', epoch,'w = ',w,'loss = ',cost_val)
print('Predict (after learning)',4,forward(4))
"""
"""随机梯度递减"""
"""
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]

w = 1.0

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

def grad(x,y):
“梯度斜率计算函数”
    grad = 2 * x * (x * w - y)
    return grad

print('Predict (before learning)',4,forward(4))
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        grad_val = grad(x,y)
        w -= 0.01 * grad_val
        print("\tgrad: ",x,y,grad)
        loss_val = loss(x, y)
    print('\t',"Epoch = ",epoch,"w = ",w,"loss_val = ",loss_val)
print('Predict (after learning)',4,forward(4))
"""

"""反向传播"""
"""
import numpy as np
import torch
import matplotlib.pyplot as plt

x_data = [1.0,2.0,3.0]
y_data = [2.0,4.0,6.0]
"tensor变量可以是点，线，面，三维变量"
w = torch.tensor([1.0])
""创建的tensor默认的不会计算梯度，所以要写一个需要梯度""
w.requires_grad = True

def forward(x):
    return x * w

def loss(x,y):
    y_pred = forward(x)
    return (y_pred - y)**2

print("Predict (before learning)",4,forward(4).item())
"item()用来将tensor转化为python的标量"
for epoch in range(100):
    for x,y in zip(x_data,y_data):
        l = loss(x,y)
        l.backward()
        print('\t\n',x,y,w.grad.item())
        w.data = w.data - 0.01*w.grad.data
        w.grad.data.zero_()
    print("process",epoch,l.item())
print("Predict (after learning)",4,forward(4).item())

plt.plot(l.item(),w.data.item())
plt.show()
"""

"""Pytorch"""
"""
import  torch

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[2.0],[4.0],[6.0]])

class LinearModle(torch.nn.Module):    #继承自Module

    def __init__(self):  #__表示构造
        super(LinearModle,self).__init__()  #调用父类的构造
        self.linear = torch.nn.Linear(1,1) #（1，1）分别对应y_pred与输入x的维度
        ""Linear(input,output,bias = True)
        其中，input与output分别对应输入与输出的维度，bias是一个bool型变量，表示是否加入偏置量
        ""

    def forward(self,x):
        y_pred = self. linear(x)
        return y_pred

modle = LinearModle()

criterion = torch.nn.MSELoss(size_average = False)
""MSELoss是Module自带的类，求损失值，size_average表示是否对Loss求均值""
optimizer = torch.optim.SGD(modle.parameters(),lr = 0.01)
""optim表示优化，torch.optim.SGD表示优化模块。
parameters用来搜索linear中表示权重的量，
lr是梯度学习率""

for epoch in range(10000):
    y_pred = modle(x_data)   #直接调用LinearModle中的forward函数
    loss = criterion(y_pred,y_data)
    print(epoch,loss)

    optimizer.zero_grad()   #清零
    loss.backward()         #反馈
    optimizer.step()        #更新

print('w = ',modle.linear.weight.item())  #调用modle中的linear中的weight
print('b = ',modle.linear.bias.item())

x_test = torch.Tensor([[4]])
y_test = modle(x_test)
print('y_pred = ',y_test.data)
"""

"""import torch
import torch.nn.functional as F

x_data = torch.Tensor([[1.0],[2.0],[3.0]])
y_data = torch.Tensor([[0],[0],[1]])

class LogisticRegressinoModle(torch.nn.Module):

    def __init__(self):
        super(LogisticRegressinoModle,self).__init__()
        self.linear = torch.nn.Linear(1,1)

    def forward(self,x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred

modle = LogisticRegressinoModle()

criterion = torch.nn.BCELoss(size_average = False)
optimizer = torch.optim.SGD(modle.parameters(),lr = 0.01)

for epoch in range(100):
    y_pred = modle(x_data)
    loss = criterion(y_pred,y_data)
    print(epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w = ',modle.linear.weight.item())
print('b = ',modle.linear.bias.item())

x_text = torch.tensor([[4.0]])
y_pred = modle(x_text)
print('y_pred = ',y_pred.data)
"""
"""
import torch
import numpy as np
xy = np.loadtxt('diabetes_data.csv',delimiter=',',dtype=np.float32)
x_data = torch.from_numpy(xy[:,:-1])
y_data = torch.from_numpy(xy[:,[-1]])

class Modle(torch.nn.Module):

    def __init__(self):
        super(Modle,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

modle = Modle()

criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(modle.parameters(),lr = 0.01)

for epoch in range(100):
    y_pred = modle.forward(x_data)
    loss = criterion(y_pred,y_data)
    print('\t',epoch,loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
"""
"""
import numpy as np
import torch
from torch.utils.data import Dataset ,DataLoader

class diabetesdata(Dataset):

    def __init__(self,filepath):
        xy = np.loadtxt(filepath,delimiter=',',dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:,:-1])
        self.y_data = torch.from_numpy(xy[:,[-1]])

    def __getitem__(self, index):
        return self.x_data[index],self.y_data[index]

    def __len__(self):
        return self.len

dataset = diabetesdata('diabetes_data.csv')
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2
                          )
class Modle(torch.nn.Module):

    def __init__(self):
        super(Modle,self).__init__()
        self.linear1 = torch.nn.Linear(8,6)
        self.linear2 = torch.nn.Linear(6,4)
        self.linear3 = torch.nn.Linear(4,1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

modle = Modle()

criterion = torch.nn.BCELoss(reduction="mean")
optimizer = torch.optim.SGD(modle.parameters(),lr = 0.01)

if __name__ =='__main__':
    for epoch in range(100):
        for i,(inputs,labels) in enumerate(train_loader,0):
            y_pred = modle(inputs)
            loss = criterion(y_pred,labels)
            print(epoch,i,loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
"""
"""
import numpy as np    #处理数组
import torch
from torch.utils.data import DataLoader  #数据加载的包
from torchvision import transforms       #图像数据原始处理的工具
from torchvision import datasets          #MNIST数据集的加载
import torch.optim as optim  #优化器的包
import torch.nn.functional as F       #神经网络中的函数式操作

batch_size = 64   #每次迭代的样本数量
transform = transforms.Compose([
    transforms.ToTensor(),      #将我们输入的图像转变成pytorch里面的张量
    transforms.Normalize((0.1307,),(0.3081,))   #两个参数分别对应均值和标准差
])

#将图像转化为张量并归一化
train_dataset = datasets.MNIST(root='../dataset/mnist/',
                               train=True,    #
                               download = True,
                               transform = transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='../dataset/mnsit/',
                              train=False,
                              download=True,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle = False,
                         batch_size=batch_size)

#定义了一个神经网络类Net，继承自torch.nn.Module，包含五个全连接层
class Net(torch.nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.l1 = torch.nn.Linear(784,512)
        self.l2 = torch.nn.Linear(512,256)
        self.l3 = torch.nn.Linear(256,128)
        self.l4 = torch.nn.Linear(128,64)
        self.l5 = torch.nn.Linear(64,10)

    def forward(self,x):
        x = x.view(-1,784)
        #将x展平，此处x的原始维度为思维[batch_size,channels,height,width]
        #view展开后，变为二维，第一维度为批次中样本数目（-1 表示自动计算这个维度的大小）
        #第二个表示展开后的向量长度（图像的总像素数目）
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

modle = Net()

criterion = torch.nn.CrossEntropyLoss()  #定义了一个新的损失函数，交叉熵损失
optimizer = torch.optim.SGD(modle.parameters(),lr = 0.01,momentum=0.5) #momentum表示动量，引入来优化训练过程

def train(epoch):    # 定义一轮训练函数
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,0):
        inputs,target = data
        optimizer.zero_grad()

        # 前馈
        outputs = modle(inputs)

        #损失计算
        loss = criterion(outputs,target)

        # 反馈
        loss.backward()

        # 参数更新
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 ==299:   #每300轮输出一次
            print('[%d,%5d] loss: %.3f'%(epoch +1,batch_idx +1,running_loss/300))
            running_loss =0.0

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            outputs = modle(images)
            _,predicted = torch.max(outputs.data,dim=1)  #dim表示取第一维度，即矩阵第一行
            total += labels.size(0)
            correct +=(predicted == labels).sum().item()
        print('Accurcy on test set: %d %%'%(100*correct/total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test()
"""
"""
import torch
in_channels,out_channels = 5,10
width,height =100,100
kernel_size  = 3
batch_size = 1

input = torch.randn(batch_size,in_channels,width,height)

conv_layer = torch.nn.Conv2d(in_channels,out_channels,kernel_size = kernel_size)

output = conv_layer(input)
print(input.shape)
print(output.shape)
print(conv_layer.weight.shape)
"""
"""
import  torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,),(0.3081,))])
train_datasets = datasets.MNIST(root='../datasets/mnist/',
                                train=True,
                                download=True,
                                transform=transform)
train_loader = DataLoader(train_datasets,
                          shuffle=True,
                          batch_size=batch_size)

test_datasets = datasets.MNIST(root='../datasets/mnist/',
                               train=False,
                               download=False,
                               transform=transform)
test_loader = DataLoader(test_datasets,
                         shuffle=False,
                         batch_size=batch_size)

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,10,kernel_size = 5)
        self.conv2 = torch.nn.Conv2d(10,20,kernel_size= 5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320,10)

    def forward(self,x):
        batch_size = x.size(0)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size,-1)
        x = self.fc(x)
        return x

model = Net()
device = torch.device("cuda:0"if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(),lr = 0.01,momentum = 0.5)

def train(epoch):
    running_loss = 0.0
    for batch_idx,data in enumerate(train_loader,1):
        inputs,labels = data
        #inputs,labels = inputs.to(device),labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx%300 == 299:
            print("[%d,%5d] loss:%.3f"%(epoch+1,batch_idx + 1,running_loss/2000))

def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images,labels = data
            images,labels = images.to(device),labels.to(device)
            outputs = model(images)
            _,predicted = torch.max(outputs.data,dim = 1 )
            total +=labels.size(0)
            correct +=(predicted == labels).sum().item()
        print('Accurcy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
   for epoch in range(10):
       train(epoch)
       test()
       
"""
"""
import  torch
import numpy as np
import torch.nn.functional as F
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim

class InceptionA(torch.nn.Module):

    def __init__(self,in_channels):
        super(InceptionA,self).__init__()
        self.branch1x1 = torch.nn.Conv2d(in_channels,16,kernel_size = 1)

        self.branch5x5_1 = torch.nn.Conv2d(in_channels,16,kernel_size = 1)
        self.branch5x5_2 = torch.nn.Conv2d(16,24,kernel_size = 5,padding = 2)

        self.branch3x3_1 = torch.nn.Conv2d(in_channels,16,kernel_size=1)
        self.branch3x3_2 = torch.nn.Conc2d(16,24,kernel_size=3,padding=1)
        self.branch3x3_3 = torch.nn.Conv2d(24,24,kernel_size=3,padding=1)

        self.branch_pool = torch.nn.Conv2d(in_channels,24,kernel_size=1)

    def forward(self,x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(x)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(x)
        branch3x3 = self.branch3x3_3(x)

        branch_pool = F.avg_pool2d(x,kernel_size=3,stride = 1,padding = 1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1,branch3x3,branch5x5,branch_pool]
        return torch.cat(outputs,dim = 1)

class ResidualBlock(torch.nn.Module):       #定义残差快，避免梯度消失

    def __init__(self,channels):
        super(ResidualBlock,self).__init__()
        self.channels = channels
        self.conv1 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding = 1)
        self.conv2 = torch.nn.Conv2d(channels,channels,kernel_size=3,padding = 1)

    def forward(self,x):
        y = F.relu(self.conv1(x))
        y = self.conv2(y)
        return F.relu(x+y)
    #return F.relu(x + self.conv2(F.relu(self.conv1(x))))

class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = torch.nn.Conv2d(1,16,kernel_size=5)
        self.conv2 = torch.nn.Conv2d(16,32,kernel_size=5)
        self.mp = torch.nn.MaxPool2d(2)

        self.rblock1 = ResidualBlock(16)
        self.rblock2 = ResidualBlock(32)

        self.fc = torch.nn.Linear(512,10)

    def forward(self,x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size,-1)
        x = self.fc(x)
        return x
"""

import torch


batch_size = 1
hidden_size = 2
input_size = 4
idx2char =['e','h','l','o']
x_data = [1,0,2,2,3]
y_data = [3,1,2,3,2]

one_hot_lookup = [
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
]

x_one_hot = [one_hot_lookup[x] for x in x_data]

inputs = torch.Tensor(x_one_hot).view(-1,batch_size,input_size)
labels = torch.LongTensor(y_data).view(-1,1)

class Model(torch.nn.Module):

    def __init__(self,input_size,hidden_size,batch_size):
        super(Model,self).__init__()
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size =hidden_size
        self.rnncell = torch.nn.RNNCell(input_size = self.input_size,
                                        hidden_size = self.hidden_size)

        def forward(self,input,hidden):
            hidden = self.rnncell(input,hidden)
            return hidden

        def init_hidden(self):
            return torch.zeros(self.batch_size,self.hidden_size)

net = Model(input_size,hidden_size,batch_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr = 0.01)
for epoch in range(15):
    loss = 0
    optimizer.zero_grad()
    hidden = net.init_hidden()
    print('Predicted string:',end='')
    for input,label in zip(inputs,labels):
        hidden = net(input,hidden)
        loss += criterion(hidden,label)
        _,idx = hidden.max(dim=1)
        print(idx2char[idx.item()],end = '')
        loss.backward()
        optimizer.step()
        print(',Epoch[%d/15] loss=%.item'%(epoch+1,loss.item()))

"""
class Model(torch.nn.Module):

    def __init__(self,input_size,hidden_size,batch_size,num_layers=1):
        super(Model,self).__init__()
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.input_size = input_size
        self.hidden_size =hidden_size
        self.rnncell = torch.nn.RNNCell(input_size = self.input_size,
                                        hidden_size = self.hidden_size,
                                        num_layers = num_layers)

        def forward(self,input,hidden):
            hidden = torch.zeros(self.num_layers,
                                  self.batch_size,
                                  self.hidden_size)
            out,_ =self.rnn(input,hidden)
            return out.view(-1,self.hidden_size)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)              #如果采用系统定义的RNN模型，代码和原来基本没什么改动
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

_, idx = outputs.max(dim=1)
idx = idx.data.numpy()
print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
print(', Epoch [%d/15] loss = %.3f' % (epoch + 1, loss.item()))"""








































































































































































