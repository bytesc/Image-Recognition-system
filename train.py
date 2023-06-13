import torch
from torch.utils.data import DataLoader
from torch import nn, optim
# optim 是 PyTorch 中的优化器模块，提供了各种优化算法来更新模型的参数
# nn 是 PyTorch 中的神经网络模块，提供了各种预定义的神经网络层和损失函数。

import datasets
import model

myModel = model.ClassificationModel3D()
# 实例化自定义的 3D 分类模型
print(torch.cuda.is_available())
# 打印是否有可用的 CUDA 设备
if torch.cuda.is_available():
    myModel = myModel.cuda()
# 如果有可用的 CUDA 设备，则将模型转移到 CUDA 上

train_set, val_set = datasets.get_datasets('./lqdata/data')
# 加载训练集和验证集
print(train_set[0][0].shape)
# 打印第一个样本的形状
train_loader = DataLoader(train_set, batch_size=1, num_workers=0, shuffle=True)
val_loader = DataLoader(val_set, batch_size=1, num_workers=0, shuffle=True)
# 定义训练集和验证集的 DataLoader，用于批量加载数据


loss_fn = nn.CrossEntropyLoss()
# 定义交叉熵损失函数
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()
# 如果有可用的 CUDA 设备，则将损失函数转移到 CUDA 上

lr = 1e-4
wd = 1e-5
# 定义学习率、权重衰减系数
optim = optim.Adam(myModel.parameters(), lr=lr, weight_decay=wd)
# 定义优化器，使用 Adam 算法，更新模型的参数

with open('logs.csv', 'a', encoding='utf-8') as f:
    f.write("\ntrain_loss,test_loss,total_accuracy,total_accuracy_for_AD\n")
# 打开一个文件，用于记录训练过程中的指标值，并写入表头

train_step = 0
test_step = 0
epoch = 3000

# 定义迭代轮数

for i in range(epoch):
    train_loss = 0
    print("------{}------".format(i))
    myModel.train()
    # 进入训练模式，启用 Dropout 和 BatchNormalization 等模型的训练模式
    for imgs, targets in train_loader:
        if torch.cuda.is_available():
            imgs = imgs.cuda()
            targets = targets.cuda()
        # 如果有可用的 CUDA 设备，则将数据转移到 CUDA 上
        output = myModel(imgs)
        # 前向传播，获取模型的输出
        optim.zero_grad()
        # 优化器梯度清零
        loss = loss_fn(output, targets)
        # 计算损失
        loss.backward()
        # 反向传播算梯度
        optim.step()
        # 优化器优化模型
        train_loss += loss.item()
        train_step += 1
        # 累加每一批次的训练损失和训练步数
    print("running_loss: ", train_loss/len(train_set))

    # 打印训练损失
    test_loss = 0
    total_accuracy = 0
    total_accuracy2 = 0
    # 初始化验证集总损失、总精度和总 AD 精度
    with torch.no_grad():
        myModel.eval()
        # 进入评估模式，关闭 Dropout 和 BatchNormalization 等模型的训练模式
        for imgs, targets in val_loader:
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            # 如果有可用的 CUDA 设备，则将数据转移到 CUDA 上
            output = myModel(imgs)
            # 前向传播，获取模型的输出
            loss = loss_fn(output, targets)
            # 计算损失
            test_loss += loss.item()
            # 累加每一个样本的验证损失
            accuracy = (output.argmax(1) == targets)
            accuracy = accuracy.sum()
            # 计算总精度
            accuracy2 = 0
            p = 0
            # 计算 AD 精度
            for j in output.argmax(1):
                if j.item() != 0 and targets[p] != 0:
                    accuracy2 += 1
                if j.item() == 0 and targets[p] == 0:
                    accuracy2 += 1
                p += 1
            # 判断输出类别是否为 0，以及目标类别是否为 0，如果都是则认为是 AD 类别预测正确
            total_accuracy += int(accuracy.item())
            total_accuracy2 += int(accuracy2)
            # 累加总精度和总 AD 精度
            test_step += 1

    print("test_loss: ", test_loss/len(val_set))
    # 打印验证损失
    print("total_accuracy: ", total_accuracy/len(val_set))
    print("total_accuracy_for_AD: ", total_accuracy2 / len(val_set))
    # 打印总精度和总 AD 精度
    torch.save(myModel, "./model_save/myModel_{}.pth".format(i))
    # 保存模型参数
    with open('logs.csv', 'a', encoding='utf-8') as f:
        data = str(train_loss / len(train_set)) + "," + str(test_loss / len(val_set)) + "," + \
               str(total_accuracy / len(val_set)) + "," + str(total_accuracy2 / len(val_set)) + ",\n"
        f.write(data)
    # 将训练过程中的指标值写入文件
