import argparse
from train import train
from eval import test
import os

os.environ['TOKENIZERS_PARALLELISM'] = 'false'  # 或者 'true'，根据你的需要

# 主函数
def main(args):
    if args.mode == 'train':
        train(args)
    elif args.mode == 'test':
        test(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #模型参数
    parser.add_argument('--num_loops', default=2, type=int)
    parser.add_argument('--N', default=22, type=int)
    parser.add_argument('--M', default=33, type=int)
    parser.add_argument('--lora_rank', default=10, type=int)
    parser.add_argument('--class_num', default=2, type=int)

    #训练参数
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--epoches', default=5, type=int)
    parser.add_argument('--resume', default='', type=str)

    #基础配置
    parser.add_argument('--task', default='PStance', type=str)
    parser.add_argument('--mode', default='train', choices=['train','test'], type=str)
    parser.add_argument('--save_dir', default='results', type=str)
    parser.add_argument('--val_frequence', default=1, type=int)
    parser.add_argument('--model_id', default='/root/autodl-tmp/LLMs/Llama-2-7b-chat-hf', type=str)
    parser.add_argument('--seqlen', default=128, type=int)
    parser.add_argument('--device', default='cuda:0', type=str)
    parser.add_argument('--print_frequence', default=50, type=int)


    args = parser.parse_args()
    main(args)