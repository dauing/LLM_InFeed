from sklearn.metrics import f1_score, accuracy_score
from collections import defaultdict
import torch
from tqdm import tqdm

def calculate_f_metrics(y_true, y_pred, targets=None, num_classes=None, average_by_target=False, task_type=None):
    """
    计算F指标的通用函数

    参数:
        y_true: 真实标签
        y_pred: 预测标签
        targets: 目标列表(用于按目标平均的情况)
        num_classes: 类别数量
        average_by_target: 是否按目标平均计算F指标
        task_type: 任务类型(可选，用于特殊处理)

    返回:
        f_macro: 宏观F1分数
        favg_scores: 各目标的F1分数(如果按目标平均)
    """
    f_macro_scores = {}

    if average_by_target and targets is not None:
        # 按目标分组计算F1
        target_groups = defaultdict(lambda: {'y_true': [], 'y_pred': []})
        for true, pred, target in zip(y_true, y_pred, targets):
            target_groups[target]['y_true'].append(true)
            target_groups[target]['y_pred'].append(pred)

        for target, data in target_groups.items():
            y_true_target = data['y_true']
            y_pred_target = data['y_pred']

            if task_type == "PStance" and num_classes == 2:
                # 二分类任务的特殊处理
                f1_0 = f1_score(y_true_target, y_pred_target, pos_label=0, zero_division=0)
                f1_1 = f1_score(y_true_target, y_pred_target, pos_label=1, zero_division=0)
                f_macro = (f1_0 + f1_1) / 2
            else:
                # 多分类任务的常规处理
                f_macro = f1_score(y_true_target, y_pred_target, average='macro', zero_division=0)

            f_macro_scores[target] = f_macro
            print(f"Favg for target '{target}': {f_macro:.4f}")

        # 计算最终的宏观F1
        favg = sum(f_macro_scores.values()) / len(f_macro_scores)
    else:
        # 不按目标平均，直接计算宏观F1
        favg = f1_score(y_true, y_pred, average='macro', zero_division=0)

    return favg

def val(model, dataloader, loss_fn, args):
    total_pred, total_y, total_targets = [[] for i in range(args.num_loops)], [], []

    for batch_i, batch_loader in enumerate(tqdm(dataloader)):

        inputs, labels, targets = batch_loader
        total_targets.extend(targets)


        with torch.no_grad():
            outputs = model(inputs, args.seqlen, args.device)["logits"]

        for i in range(args.num_loops):
            pred_y = torch.max(outputs[i], 1).indices
            total_pred[i].append(pred_y.cpu())

        total_y.append(labels.cpu())

    total_y = torch.cat(total_y)
    total_pred = [torch.cat(total_pred[i]) for i in range(args.num_loops)]
    for i in range(args.num_loops):
        acc = accuracy_score(total_y, total_pred[i])
        print(f"Accuracy at step {i+1}: {acc:.4f}")

        # 计算F指标
        favg = calculate_f_metrics(
            y_true=total_y,
            y_pred=total_pred[i],
            targets=total_targets if args.task in ["PStance", "SE16T6A"] else None,
            num_classes=2 if args.task == "PStance" else None,
            average_by_target=args.task in ["PStance", "SE16T6A"],
            task_type=args.task
        )

        print(f"Final Fmacro: {favg:.4f}")
    return acc, favg