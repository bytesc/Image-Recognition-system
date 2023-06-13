import os
import nibabel as nib  # 用于读取NIfTI格式文件的Python库
import torch
from torch.utils.data import Dataset  # Dataset是PyTorch中用于表示数据集的抽象类
from torch.utils.data import DataLoader  # DataLoader是PyTorch中用于加载数据集的工具类

LABEL_LIST = ["AD", "CN", "MCI", "EMCI", "LMCI"]  # 定义标签列表


class MyData(Dataset):
    # 自定义数据集类，继承自Dataset
    def __init__(self, root_dir, label_dir):
        # 初始化函数，定义类的成员变量
        self.root_dir = root_dir  # 数据集根目录
        self.label_dir = label_dir  # 数据集标签子目录
        self.path = os.path.join(self.root_dir, self.label_dir)  # 根据根目录和标签子目录，获取数据集文件路径
        self.img_path = os.listdir(self.path)  # 获取数据集路径下的文件名列表

    def __getitem__(self, idx):
        # 获取数据集中的某一条数据
        img_name = self.img_path[idx]  # 获取文件名列表中的第idx个文件名
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)  # 获取数据集文件的路径

        img = nib.load(img_item_path)  # 使用nibabel库读取NIfTI格式文件
        img = img.get_fdata()  # 获取NIfTI图像的数据

        img = torch.from_numpy(img)  # 将图像数据转换为PyTorch张量
        img = img.squeeze()  # 去除张量中维数为1的维度
        img = img.reshape(1, -1, 256, 256)  # 根据torch.nn.Conv3d的输入要求(C, D, H, W)，对数据进行reshape
        img = img[:, 0:160, :, :].float()  # 对数据进行裁剪和类型转换

        label = self.label_dir  # 获取当前数据的标签
        label = LABEL_LIST.index(label)  # 将标签转换为数字编码
        label = torch.tensor(label)  # 将标签转换为PyTorch张量
        return img, label  # 返回数据和标签的元组

    def __len__(self):
        # 获取数据集的长度（即数据集中的样本数）
        return len(self.img_path)  # 返回文件名列表的长度


def get_datasets(myroot_dir="./data"):
    # 获取数据集
    AD_dir = "AD"  # AD标签的子目录名
    CN_dir = "CN"  # CN标签的子目录名
    MCI_dir = "MCI"  # MCI标签的子目录名
    EMCI_dir = "EMCI"  # EMCI标签的子目录名
    LMCI_dir = "LMCI"  # LMCI标签的子目录名

    # 获取训练集数据集对象
    ad_dataset = MyData(myroot_dir, AD_dir)  # 获取AD标签子目录下的数据集对象
    cn_dataset = MyData(myroot_dir, CN_dir)  # 获取CN标签子目录下的数据集对象
    mci_dataset = MyData(myroot_dir, MCI_dir)  # 获取MCI标签子目录下的数据集对象
    emci_dataset = MyData(myroot_dir, EMCI_dir)  # 获取EMCI标签子目录下的数据集对象
    lmci_dataset = MyData(myroot_dir, LMCI_dir)  # 获取LMCI标签子目录下的数据集对象

    # 获取测试集数据集对象
    myroot_dir += "/test"  # 将数据集根目录路径更新为测试集根目录路径
    ad_dataset_test = MyData(myroot_dir, AD_dir)  # 获取测试集AD标签子目录下的数据集对象
    cn_dataset_test = MyData(myroot_dir, CN_dir)  # 获取测试集CN标签子目录下的数据集对象
    mci_dataset_test = MyData(myroot_dir, MCI_dir)  # 获取测试集MCI标签子目录下的数据集对象
    emci_dataset_test = MyData(myroot_dir, EMCI_dir)  # 获取测试集EMCI标签子目录下的数据集对象
    lmci_dataset_test = MyData(myroot_dir, LMCI_dir)  # 获取测试集LMCI标签子目录下的数据集对象

    # 将所有数据集对象相加，得到训练集和测试集的数据集对象
    train_set = ad_dataset + cn_dataset + lmci_dataset + mci_dataset + emci_dataset  # 将AD、CN、LMCI、MCI、EMCI训练集数据集对象相加
    val_set = ad_dataset_test + cn_dataset_test + lmci_dataset_test + mci_dataset_test + emci_dataset_test  # 将测试集AD、CN、LMCI、MCI、EMCI数据集对象相加

    return train_set, val_set  # 返回训练集和测试集的数据集对象


if __name__ == "__main__":
    DATA_PATH = './lqdata/data'
    train_set, val_set = get_datasets(DATA_PATH)  # 获取训练集和验证集

    print(train_set[0][0].shape)  # 输出第一张图像的形状

    train_loader = DataLoader(train_set, batch_size=1, num_workers=0, shuffle=True)  # 创建训练集DataLoader，设置批大小、工作进程数和是否乱序
    val_loader = DataLoader(val_set, batch_size=1, num_workers=0, shuffle=True)  # 创建验证集DataLoader，设置批大小、工作进程数和是否乱序

    for imgs, targets in train_loader:
        # 遍历训练集DataLoader
        print(imgs.shape)  # 输出当前批次的图像形状
        print(LABEL_LIST[targets.item()])  # 输出当前批次的标签