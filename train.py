import time
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm 
from data import *
from model import RecurrentLlama
import torch.nn as nn
from valid import val
import torch
from torch.utils.data import DataLoader, Sampler
from torch.utils.tensorboard import SummaryWriter
from utils import *

def train(args):
    # 加载模型
    model = RecurrentLlama(N=args.N,M=args.M,model_id=args.model_id,device=args.device,num_loops=args.num_loops,num_classes=args.class_num)
    model.to(args.device)
    model.to(torch.float32)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Gradient of {name}")
    
    # 加载数据
    if args.task == 'PStance':
        dataset_train = PStance(csv_path="/root/llm-main/datasets/P-Stance",mode_list=["train"])
        dataset_test = PStance(csv_path="/root/llm-main/datasets/P-Stance",mode_list=["test"])
    elif args.task == 'VAST':
        dataset_train = VAST(csv_path="/root/llm-main/datasets/VAST",mode="train")
        dataset_test = VAST(csv_path="/root/llm-main/datasets/VAST",mode="test")
    elif args.task == 'SE16T6A':
        dataset_train = SE16T6A(path="/root/llm-main/datasets/SE16T6A",mode="train")
        dataset_test = SE16T6A(path="/root/llm-main/datasets/SE16T6A",mode="test")
    
    dataloader_train = DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,drop_last=True)
    dataloader_test = DataLoader(dataset_test, batch_size=args.batch_size, shuffle=True,drop_last=True)

    #记录日志
    # log_dir = f"runs/{time.strftime('%Y%m%d-%H%M%S')}"
    time_stamp = time.strftime("%Y%m%d-%H%M%S")
    save_dir = f"results/{time_stamp}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    log_dir = "/root/tf-logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir)

    # 训练工具
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoches, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss().to(args.device)
    print_loss = 0.0
    best_acc = 0.0
    best_f1 = 0.0
    # powers = [1.0/(args.num_loops - i) for i in range(args.num_loops)]
    powers = []
    for i in range(args.num_loops-1):
        powers.append(0.1)
    powers.append(1.0)
    print(f"powers: {powers}")
    now_epoch = 0

    #断点续训
    if args.resume != '':
        now_epoch = model.load_adapter_head(args.resume, 'model.pth')
        optimizer.load_state_dict(torch.load(args.resume + '/optimizer.pth',weights_only=True))
        now_epoch += 1

    loss_single = [0.0 for i in range(args.num_loops)]

    for epoch in range(now_epoch, args.epoches):
        print(f'--------------------------- epoch {epoch} ---------------------------')
        acc_list = []
        for batch_i, batch in enumerate(tqdm(dataloader_train)):
            if args.task == 'PStance' or args.task == 'SE16T6A':
                sents, labels, _ = batch
            elif args.task == 'VAST':
                sents, labels = batch
            labels = labels.to(args.device)
            outputs_all = model.forward(sents, args.seqlen, args.device)
            outputs = outputs_all["logits"]

            pred_y = torch.max(outputs[args.num_loops-1], 1).indices
            acc = accuracy_score(labels.cpu(), pred_y.cpu())
            acc_list.append(acc)

            loss_all = None
            for i in range(args.num_loops):
                loss = criterion(outputs[i], labels)
                loss_single[i] += loss.item()
                if i == 0:
                    loss_all = loss*powers[i]
                else:
                    loss_all += loss*powers[i]
            loss_all.backward()
            optimizer.step()
            optimizer.zero_grad()
            print_loss += loss_all.item()
            

            torch.cuda.empty_cache()
            
            if (batch_i + 1)%args.print_frequence == 0:
                writer.add_scalar('loss', print_loss/args.print_frequence, epoch * len(dataloader_train) + batch_i + 1)
                time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                print(f'Epoch: {epoch}  Time: {time_now}  LR: {scheduler.get_lr()}  Loss: {print_loss:.4f}')
                print_loss = 0.0
        
        print(f'acc: {np.mean(acc_list):.4f}')
        writer.add_scalar('metrics/trainAcc', np.mean(acc_list), epoch)
        if epoch % args.val_frequence == 0:
            print(f'valid epoch {epoch}')
            val_acc, val_f1 = val(model, dataloader_test, criterion, args=args)
            if val_acc > best_acc:
                best_acc = val_acc
            if val_f1 > best_f1:
                best_f1 = val_f1
                model.save_model(save_dir,'best.pth',epoch)#保存最佳模型
            writer.add_scalar('metrics/valAcc', val_acc, epoch)
            writer.add_scalar('metrics/valF1', val_f1, epoch)
            print("val acc: ",val_acc)
            print("val f1: ",val_f1)
        #保存最新模型及优化器
        model.save_model(save_dir,'model.pth',epoch)
        torch.save(optimizer.state_dict(), save_dir + '/optimizer.pth')
        scheduler.step()



