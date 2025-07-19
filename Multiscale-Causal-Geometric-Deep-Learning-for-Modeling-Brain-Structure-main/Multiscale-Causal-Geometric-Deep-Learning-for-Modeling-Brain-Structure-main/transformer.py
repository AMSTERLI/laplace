import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool


class SingleGraphTransformer(nn.Module):
    """独立单图Transformer模块（处理sMRI/dMRI）"""

    def __init__(self, input_dim, hidden_dim=64, nhead=8, num_layers=4):
        super().__init__()

        # 特征编码层
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=4 * hidden_dim,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers)

        # 回归预测头
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, data):
        """
        输入:
            data: Data对象包含:
                x: (num_nodes, input_dim)
                edge_index: (2, num_edges)
                batch: (num_nodes,)  # 用于图级别任务
        输出:
            pred: (batch_size, 1)
        """
        # 维度验证
        assert data.x.dim() == 2, f"输入特征必须是二维张量，当前维度：{data.x.dim()}"
        expected_features = self.encoder[0].in_features
        assert data.x.size(1) == expected_features, (
            f"特征维度不匹配！输入维度：{data.x.size(1)}，"
            f"模型预期：{expected_features}"
        )

        # 特征编码
        x = self.encoder(data.x)  # (num_nodes, hidden_dim)

        # Transformer处理
        x = x.unsqueeze(1)  # (num_nodes, 1, hidden_dim)
        x = self.transformer(x)
        x = x.squeeze(1)  # (num_nodes, hidden_dim)

        # 图级别池化
        pooled = global_mean_pool(x, data.batch)  # (batch_size, hidden_dim)

        # 回归预测
        return self.regressor(pooled)  # (batch_size, 1)


class JointGraphTransformer(nn.Module):
    """联合双Transformer预测模块"""

    def __init__(self, sMRI_transformer, dMRI_transformer, fusion_dim=128):
        super().__init__()
        # 固定预训练参数
        self.sMRI_transformer = sMRI_transformer
        self.dMRI_transformer = dMRI_transformer
        for param in self.sMRI_transformer.parameters():
            param.requires_grad = False
        for param in self.dMRI_transformer.parameters():
            param.requires_grad = False

        # 动态特征融合
        self.fusion = nn.Sequential(
            nn.Linear(2, fusion_dim),  # 两个单模预测结果拼接
            nn.LayerNorm(fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 1)
        )

    def forward(self, sMRI_data, dMRI_data):
        """
        输入:
            sMRI_data: Data对象包含sMRI图
            dMRI_data: Data对象包含dMRI图
        输出:
            joint_pred: (batch_size, 1)
        """
        # 单模预测
        sMRI_pred = self.sMRI_transformer(sMRI_data)  # (bs,1)
        dMRI_pred = self.dMRI_transformer(dMRI_data)  # (bs,1)

        # 特征拼接
        combined = torch.cat([sMRI_pred, dMRI_pred], dim=1)  # (bs,2)

        # 联合预测
        return self.fusion(combined)  # (bs,1)


class EnhancedJointTransformer(nn.Module):
    """增强版特征级融合（替代方案）"""

    def __init__(self, sMRI_transformer, dMRI_transformer, hidden_dim=128):
        super().__init__()
        self.sMRI_transformer = sMRI_transformer
        self.dMRI_transformer = dMRI_transformer

        # 特征融合模块
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # 假设单模输出hidden_dim
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def get_graph_embeddings(self, data):
        """获取中间特征表示"""
        # sMRI特征提取
        sMRI_feat = self.sMRI_transformer.encoder(data.x)
        sMRI_feat = global_mean_pool(sMRI_feat, data.batch)

        # dMRI特征提取
        dMRI_feat = self.dMRI_transformer.encoder(data.x)
        dMRI_feat = global_mean_pool(dMRI_feat, data.batch)

        return sMRI_feat, dMRI_feat

    def forward(self, sMRI_data, dMRI_data):
        # 获取特征嵌入
        sMRI_emb, dMRI_emb = self.get_graph_embeddings(sMRI_data), self.get_graph_embeddings(dMRI_data)

        # 特征拼接
        combined = torch.cat([sMRI_emb, dMRI_emb], dim=1)

        # 联合预测
        return self.fusion(combined)

# ===== Added for MI calculation (monkey‑patch embed) =====
def _sgt_embed(self, x):
    """Return node embeddings after initial encoder (gradient intact)."""
    return self.encoder(x)
try:
    SingleGraphTransformer.embed
except AttributeError:
    SingleGraphTransformer.embed = _sgt_embed
