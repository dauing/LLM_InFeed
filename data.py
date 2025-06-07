import os
import pandas as pd
from torch.utils.data import Sampler
import torch

class RandomSubsetSampler(Sampler):
    def __init__(self, data_source, subset_size):
        self.data_source = data_source
        self.subset_size = subset_size

    def __iter__(self):
        # 生成不重复的随机索引
        indices = torch.randperm(len(self.data_source))[:self.subset_size]
        yield from indices.tolist()

    def __len__(self):
        return self.subset_size

class PStance:
    def __init__(self, csv_path="/home/alpha/LLM/datasets/P-Stance", mode_list=["train", "test", "val"], target_list=["trump", "biden", "bernie"]):
        self.name = "PStance"
        self.file_list = []
        for mode in mode_list:  # Avoid naming conflict with mode_list parameter
            for target in target_list:
                file_name = f"raw_{mode}_{target}.csv"
                file_path = os.path.join(csv_path, file_name)
                if os.path.exists(file_path):  # Check if file exists
                    self.file_list.append(file_path)
                else:
                    print(f"Warning: {file_path} does not exist.")

        # Read and merge all valid CSV files
        df_list = [pd.read_csv(file) for file in self.file_list]
        self.data = pd.concat(df_list, ignore_index=True) if df_list else pd.DataFrame()
        self.num_samples = len(self.data)

        self.idx = 0  # Initialize index for iteration


    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __next__(self):
        if self.idx < self.num_samples:
            result = self.__getitem__(self.idx)
            self.idx += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range for dataset.")

        row = self.data.iloc[idx]
        text = row["Tweet"]
        target = row["Target"]
        stance = row["Stance"]
        if stance == "FAVOR":
            stance = 0
        elif stance == "AGAINST":
            stance = 1
        else:
            stance = None
        text = "text: " + text + ", target: " + target
        label = stance
        return text, label, target

class VAST:#上千
    def __init__(self, csv_path="E:/Adatafile/StudyAndWork/project/zero-shot-stance-master/data/VAST", mode='train'):
        self.name = "Vast"
        if mode == 'train':
            file_name = f"vast_train.csv"
        elif mode == 'test':
            file_name = f"vast_test.csv"
        else:
            raise ValueError("Invalid mode.")
        
        file_path = os.path.join(csv_path, file_name)
        self.data = pd.read_csv(file_path)
        self.num_samples = len(self.data)
        self.idx = 0  # Initialize index for iteration


    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __next__(self):
        if self.idx < self.num_samples:
            result = self.__getitem__(self.idx)
            self.idx += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range for dataset.")

        row = self.data.iloc[idx]
        text = row["post"]
        target = row["topic"]
        text = "topic: " + target + ", text: " + text
        label = row["label"]
        label = int(label)
        if label not in [0, 1, 2]:
            print(f"Warning: label {label} is not in [0, 1, 2].")
        return text, label
    
class ICA_V1:
    def __init__(self, csv_path="E:/Adatafile/StudyAndWork/project/zero-shot-stance-master/data/ICA-V1", mode='train'):
        self.name = "ICA_V1"
        path_nosarc = os.path.join(csv_path, 'nosarc')
        list_nosarc = os.listdir(path_nosarc)
        path_sarc = os.path.join(csv_path, 'sarc')
        list_sarc = os.listdir(path_sarc)
        if mode == 'train':
            self.list_nosarc = list_nosarc[:int(len(list_nosarc)*0.8)]
            self.list_sarc = list_sarc[:int(len(list_sarc)*0.8)]
        elif mode == 'test':
            self.list_nosarc = list_nosarc[int(len(list_nosarc)*0.8):]
            self.list_sarc = list_sarc[int(len(list_sarc)*0.8):]
        else:
            raise ValueError("Invalid mode.")
        self.data = []
        self.label = []
        for file in self.list_nosarc:
            with open(os.path.join(path_nosarc, file), 'r', encoding='utf-8') as f:
                text = f.read()
            self.data.append(text)
            self.label.append(0)
        for file in self.list_sarc:
            with open(os.path.join(path_sarc, file), 'r', encoding='utf-8') as f:
                text = f.read()
            self.data.append(text)
            self.label.append(1)
        self.num_samples = len(self.data)
        self.idx = 0  # Initialize index for iteration

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __next__(self):
        if self.idx < self.num_samples:
            result = self.__getitem__(self.idx)
            self.idx += 1
            return result
        else:
            raise StopIteration

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range for dataset.")

        text = self.data[idx]
        label = self.label[idx]
        return text, label
        
class ICA_V2:
    def __init__(self, csv_path="E:/Adatafile/StudyAndWork/project/zero-shot-stance-master/data/ICA-V2", mode='train'):
        self.name = "ICA_V2"
        pass

class SE183A:#一百多
    def __init__(self, path="C:/Users/22354/Desktop/工作/数据集/讽刺检测数据集/SE18T3A", mode='train'):
        self.name = "SE183A"
        if mode == 'train':
            self.file_name = 'train.txt'
        elif mode == 'test':
            self.file_name = 'test.txt'
        with open (os.path.join(path, self.file_name), 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.data = []
        for line in lines:
            line = line.strip().split('\t')
            self.data.append(line)
        self.num_samples = len(self.data)
        self.idx = 0  # Initialize index for iteration

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range for dataset.")
        text = self.data[idx][2]
        label = self.data[idx][1]
        return text, label

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __next__(self):
        if self.idx < self.num_samples:
            result = self.__getitem__(self.idx)
            self.idx += 1
            return result
        else:
            raise StopIteration
class SST2:
    def __init__(self, csv_path="E:/Adatafile/StudyAndWork/project/zero-shot-stance-master/data/SST2", mode='train'):
        self.name = "SST2"
        pass

class SE16T6A:#170
    def __init__(self, path="C:/Users/22354/Desktop/工作/数据集/立场检测数据集/SE16T6A", mode='train'):
        self.name = "SE16T6A"
        if mode == 'train':
            self.file_name = 'train.txt'
        elif mode == 'test':
            self.file_name = 'test.txt'
        with open (os.path.join(path, self.file_name), 'r', encoding='utf-8',errors='ignore') as f:
            lines = f.readlines()
        lines = lines[1:]
        self.data = []
        for line in lines:
            line = line.strip().split('\t')
            self.data.append(line)
        self.num_samples = len(self.data)
        self.idx = 0  # Initialize index for iteration

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range for dataset.")
        text = self.data[idx][2]
        target = self.data[idx][1]
        label = self.data[idx][-1]
        if label == 'AGAINST':
            label = 0
        elif label == 'FAVOR':
            label = 1
        elif label == 'NONE':
            label = 2
        else:
            raise ValueError("Invalid label.")
        text = "Topic: " + target + ". Text: " + text
        return text, label, target

    def __iter__(self):
        for idx in range(len(self)):
            yield self.__getitem__(idx)

    def __next__(self):
        if self.idx < self.num_samples:
            result = self.__getitem__(self.idx)
            self.idx += 1
            return result


if __name__ == "__main__":
    # dataset = VAST()
    # # dataset = PStance(csv_path='C:/Users/22354/Desktop/工作/数据集/立场检测数据集/P-Stance')
    # for data in dataset:
    #     text, label = data
    #     print(len(text))
    #     # print(text)
    #     # print(label)
    # # dataset = SE183A()
    dataset = SE16T6A()
    for data in dataset:
        text, label, target = data
        print(len(text),target,label)
