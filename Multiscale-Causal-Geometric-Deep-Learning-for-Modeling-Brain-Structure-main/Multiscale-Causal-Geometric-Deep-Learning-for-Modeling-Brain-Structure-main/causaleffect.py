import numpy as np
import torch
import torch.nn as nn
from utils import calculate_conditional_MI, MI as calculate_MI
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.glob import global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.data import Data

def clear_masks(model):
    """ clear the edge weights to None """
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = False
            module.__edge_mask__ = None


def set_masks(model, edgemask):
    for module in model.modules():
        if isinstance(module, MessagePassing):
            module.__explain__ = True
            module.__edge_mask__ = edgemask

def get_readout_layers(readout):
    readout_func_dict = {
        "mean": global_mean_pool,
        "sum": global_add_pool,
        "max": global_max_pool
    }
    readout_func_dict = {k.lower(): v for k, v in readout_func_dict.items()}
    ret_readout = []
    for k, v in readout_func_dict.items():
        if k in readout.lower():
            ret_readout.append(v)
    return ret_readout


def joint_uncond(alpha, beta, data, casual_decoder, transformer, device, k):
    """
    计算因果效应，并利用 transformer 进行年龄回归预测。

    参数:
      - alpha, beta: 节点级因果潜变量（形状：[num_nodes, feature_dim]），要求未池化；
      - data: 图数据对象，包含 data.edge_index, data.batch, data.x, data.y（回归标签）；
      - casual_decoder: 因果解码器，用于生成边权掩码；
      - transformer: 用于回归预测的 GraphTransformer 模型；
      - device: 设备；
      - k: 当前训练步数（可用于分阶段操作）。

    返回:
      - CausalEffect: 计算得到的因果效应（标量）；
      - predictions: transformer 模型对输入图数据的连续预测（形状：[batch, 1]）。
    """
    # 若 data 中未包含 batch 信息，则默认所有节点属于同一图
    if not hasattr(data, 'batch') or data.batch is None:
        data.batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)

    # 利用预定义的读出层（例如均值池化）对 alpha 和 beta 进行图级池化
    readout_layers = get_readout_layers("mean")  # 需预先定义该函数，返回池化函数列表
    pooled_alpha = [readout(alpha.to(device), data.batch.to(device)) for readout in readout_layers]
    graph_alpha = torch.cat(pooled_alpha, dim=-1)
    pooled_beta = [readout(beta.to(device), data.batch.to(device)) for readout in readout_layers]
    graph_beta = torch.cat(pooled_beta, dim=-1)

    # 利用因果解码器生成边权掩码
    ax, aindex = casual_decoder(alpha.to(device))
    aedge = aindex[data.edge_index[0], data.edge_index[1]].to(device)
    aedge = torch.sigmoid(aedge)
    # 构造 dense edge_bias（假设 batch=1，此处为简单示例）
    num_nodes = data.x.size(0)
    edge_bias = torch.zeros((num_nodes, num_nodes), device=device)
    edge_bias[data.edge_index[0], data.edge_index[1]] = aedge

    # 使用整个图的节点特征进行预测
    # 假设 data.x 的形状为 (num_nodes, in_dim)，需扩展 batch 维度
    predictions = transformer(data)
    # predictions: 形状 (1, out_dim)；对于回归任务 out_dim=1
    # 计算因果效应（例如利用条件互信息）
    labels = data.y.to(device).float()  # 假设 data.y 为标量或形状 [batch]
    CausalEffect = calculate_conditional_MI(graph_alpha, labels, graph_beta)
    return CausalEffect, predictions

def get_retain_mask(drop_probs, shape, device):
    tau = 1.0
    uni = torch.rand(shape).to(device)
    eps = torch.tensor(1e-8).to(device)
    tem = (torch.log(drop_probs + eps) - torch.log(1 - drop_probs + eps) + torch.log(uni + eps) - torch.log(
        1.0 - uni + eps))
    mask = 1.0 - torch.sigmoid(tem / tau)
    return mask


def compute_alpha_gradients(alpha, logits, classifier, casual_decoder, target_class=None):
    """
    计算已经存在的 alpha 对目标输出的梯度。

    参数：
        alpha: 已计算好的潜在变量中的 α 部分（形状为 [N, Nalpha]），必须是 requires_grad 为 True。
        logits: 已经通过 classifier 得到的预测输出，形状为 [batch_size, num_classes]。
        classifier: 分类器模块，用于反向传播前梯度清零。
        casual_decoder: 因果解码器模块，也一并清零梯度（根据需要）。
        target_class: 可选，指定目标类别索引；如果为 None，则取 logits 中的最大值作为目标。

    返回：
        alpha_grad: 与 alpha 同形状的梯度张量。
    """
    # 确保 alpha 已经设置了梯度跟踪并保留梯度信息
    alpha.requires_grad_(True)
    alpha.retain_grad()

    # 构造目标标量
    if target_class is not None:
        target_score = logits[:, target_class].mean()
    else:
        target_score = logits.max()

    # 清零相关模块的梯度
    classifier.zero_grad()
    casual_decoder.zero_grad()

    # 反向传播计算目标标量对 alpha 的梯度
    target_score.backward(retain_graph=True)

    # 提取 alpha 的梯度
    alpha_grad = alpha.grad.detach().clone()

    # 清除梯度，防止后续累积
    alpha.grad.zero_()
    return alpha_grad
