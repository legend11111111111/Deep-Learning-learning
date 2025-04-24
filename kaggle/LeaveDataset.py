from torch.utils.data import Dataset, DataLoader
import os
import pandas as pd
from PIL import Image

base_path = './kaggle/input/classify-leaves'
out_path = './kaggle/working'



train_df= pd.read_csv(os.path.join(base_path,"train.csv"))
# 获取所有唯一的类别（叶子种类）
unique_labels = train_df["label"].unique()

# 创建 类别 → 索引 的映射
label2idx = {label: idx for idx, label in enumerate(unique_labels)}

# 创建反向映射（id → label）
idx2label = {v: k for k, v in label2idx.items()}


class LeaveDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_path, self.data.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)  # 应用转换

        label_name = self.data.iloc[idx, 1]
        label = label2idx[label_name]  # 转换为整数索引

        return image, label  # 返回 (图片, 标签)


class LeaveValDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_path, self.data.iloc[idx, 0])
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)  # 应用转换

        return image