import torch
from data import *
from model import RecurrentLlama
import torch.nn as nn
from valid import val
import torch
from torch.utils.data import DataLoader
def test(args):
    # 加载模型
    model = RecurrentLlama(model_id=args.model_id,device=args.device,num_loops=args.num_loops,num_classes=args.class_num)
    model.to(args.device)
    model.to(torch.float32)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Gradient of {name}")
    model.load_adapter_head(args.resume, 'best.pth')
    # 加载数据
    if args.task == 'PStance':
        dataset_test = PStance(csv_path="/root/llm-main/datasets/P-Stance",mode_list=["test"])
    elif args.task == 'VAST':
        dataset_test = VAST(csv_path="/root/llm-main/datasets/VAST",mode="test")
    elif args.task == 'SE16T6A':
        dataset_test = SE16T6A(path="/root/llm-main/datasets/SE16T6A",mode="test")
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,drop_last=False)
    
    criterion = nn.CrossEntropyLoss().to(args.device)

    model.eval()

    with torch.no_grad():
        val_acc, val_f1 = val(model, dataloader_test, criterion, args=args)
        print("val acc: ",val_acc)
        print("val f1: ",val_f1)




