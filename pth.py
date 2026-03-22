import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

training_data = datasets.MNIST(
    root="data",#文件下载位置的根目录
    train=True,
    download=True,#确认是否已经从网络上爬取到了数据集，如果已经下载到了就不必再下载了。
    transform=ToTensor(),#少了一个逗号
    #张量，图片不能直接导入神经网络，需要转换成张量
)

test_data = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),#少了一个逗号
)

#6万张图片打包
train_dataloader = DataLoader(training_data,batch_size=64)#如果电脑CPU内存不够，可以把batch_size调小一点，但是2的整数倍
#一张图片都训练不了，方法1：可以先把图片压缩，方法2：换一个GPU跑

test_dataloader = DataLoader(test_data,batch_size=64)

#测试使用什么跑
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using {device} device")

#固定的框架可以背下来
class CNN(nn.Module):
    #设置三层
    def __init__(self):
        super(CNN,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(16,32,5,1,2),
            nn.ReLU(),
            nn.Conv2d(32,32,5,1,2),

            #新增一层，Accuracy在98.7%左右，效果反而变差了
            # nn.ReLU(),
            # nn.Conv2d(64,128,5,1,2),

            nn.MaxPool2d(2),
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(32,64,5,1,2),
            nn.ReLU(),
            # nn.MaxPool2d(2),
        )

        self.out = nn.Linear(64*7*7,10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)  # 输出 (64,64, 7, 7)
        x = x.view(x.size(0), -1)  # flatten操作，结果为：(batch_size, 64 * 7 * 7)
        output = self.out(x)
        return output
#优化到正确率为99.97%

model = CNN().to(device)#模型导入到CPU中
print(model)

def train(dataloader,model,loss_fn,optimizer):
    model.train()#进入训练模式，权重参数不可修改
    #为什么每次训练完之后不会从头开始训练？而是接着上一次的继续训练。就是设置了model.train()
    batch_size_num = 1#计数训练到第几个batch包了
    for X,y in dataloader:
        #MNIST已经切割好训练集和测试集了，直接用。X，y
        X,y = X.to(device),y.to(device)#导入数据集到CPU

        pred = model.forward(X) #有时写的就是model(X)是一种简写形式，数据流方向
        loss = loss_fn(pred,y)#损失

        optimizer.zero_grad()
        loss.backward()#反馈损失值
        optimizer.step()#优化器优化权重参数

        loss = loss.item()
        # print(f"loss: {loss_value:>7f} [number:{batch_size_num}]")
        batch_size_num += 1

def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()    #进入测试模式，w值不可更改
    test_loss, correct = 0, 0 #with open()自动管理文件，不必再写close()，很方便。
    with torch.no_grad():   #一个上下文管理器，关闭梯度计算。当你确认不会调用Tensor.backward()的时候。这可以减少计算所用内存消耗。
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)#model和数据都必须传到相同的位置才可以，都是传到CPU或者GPU，不写to(device)的时候默认写把数据传入CPU，用GPU的时候特别注意
            pred = model.forward(X)
            test_loss += loss_fn(pred, y).item() #test_loss是会自动累加每一个批次的损失值
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            a = (pred.argmax(1) == y)  #dim=1表示每一行中的最大值对应的索引号，dim=0表示每一列中的最大值对应的索引号
            b = (pred.argmax(1) == y).type(torch.float)
    test_loss /= num_batches  #能来衡量模型测试的好坏。
    correct /= size  #平均的正确率
    print(f"Test result: \n Accuracy: {(100*correct)}%, Avg loss: {test_loss}")

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print(f"保存模型至 {filename}")

#添加这段、调整后段代码缩进，避免启动Flask同时也启动训练，不必等待训练完10轮再启动网页
if __name__ == "__main__":
    loss_fn = nn.CrossEntropyLoss()#交叉熵损失函数，订正试卷
    optimizer = torch.optim.Adam(model.parameters(),lr = 0.001)

# train(train_dataloader,model,loss_fn,optimizer)
# test(test_dataloader,model,loss_fn)

    epochs = 10 #到底选择多少呢？
    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    save_model(model, './mnist_cnn_model.pth')

    print("Done!")
    test(test_dataloader, model, loss_fn)