import os
import numpy as np
import scipy.io
import networkx as nx
import json
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from sklearn.metrics.pairwise import cosine_similarity
from networkx.readwrite import json_graph
from sklearn.random_projection import GaussianRandomProjection
from scipy.sparse.csgraph import laplacian
from scipy.sparse import csr_matrix

# 用于将 numpy 类型编码为 JSON 可序列化格式
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int_)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float_)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

# 为图添加谱特征（拉普拉斯本征向量）
def add_spectral_features(graph, k=3):
    A = nx.adjacency_matrix(graph, weight='weight')
    L_sym = csr_matrix(laplacian(A, normed=True))  # 归一化拉普拉斯
    eigenvalues, eigenvectors = np.linalg.eigh(L_sym.toarray())
    selected_vectors = eigenvectors[:, 1:k + 1]  # 去掉第一个本征向量（为常数）

    # 对每个本征向量做归一化
    for i in range(selected_vectors.shape[1]):
        vec = selected_vectors[:, i]
        vec = (vec - vec.min()) / (vec.max() - vec.min() + 1e-8)
        selected_vectors[:, i] = vec

    # 将谱特征拼接到节点原始特征后
    for i in graph.nodes():
        original_features = np.array(graph.nodes[i]['features'])
        spectral_features = selected_vectors[i]
        combined_features = np.concatenate([original_features, spectral_features])
        graph.nodes[i]['features'] = combined_features.tolist()

    return graph

# 用 GATConv 对图节点特征进行注意力增强
def apply_gat_attention(graph, num_features):
    edge_index = []
    edge_weight = []
    for u, v, data in graph.edges(data=True):
        edge_index.append([u, v])
        edge_index.append([v, u])  # 加入双向边
        edge_weight.extend([data['weight'], data['weight']])

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    x = torch.tensor([graph.nodes[i]['features'] for i in graph.nodes()], dtype=torch.float)

    # 定义 GAT 模型
    class GAT(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GATConv(num_features, num_features, heads=1, add_self_loops=True)

        def forward(self, x, edge_index):
            return self.conv1(x, edge_index)

    model = GAT()

    with torch.no_grad():
        output = model(x, edge_index)

    # 对 GAT 输出再次归一化
    output = (output - output.min()) / (output.max() - output.min() + 1e-8)
    for i in graph.nodes():
        graph.nodes[i]['features'] = output[i].numpy().tolist()

    return graph

# ===== 主处理流程 =====
base_folder = "vector"
output_folder = "processed_graph"

os.makedirs(output_folder, exist_ok=True)

for folder_name in os.listdir(base_folder):
    subfolder_path = os.path.join(base_folder, folder_name)
    if os.path.isdir(subfolder_path):
        mat_file_path = os.path.join(subfolder_path, "results_for_analysis.mat")
        if os.path.exists(mat_file_path):
            print(f"Processing folder: {folder_name}")
            mat_data = scipy.io.loadmat(mat_file_path)

            harmonic_cortex = mat_data['harmonic_cortex']
            value_fiber = mat_data['value_fiber'].squeeze()
            value_cortex = mat_data['value_cortex'].squeeze()

            # 降维处理（最多使用前 200000 个点）
            if harmonic_cortex.shape[0] >= 200000:
                data_for_projection = harmonic_cortex[:200000, :]
            else:
                data_for_projection = harmonic_cortex

            # 高斯随机投影降维，生成特征向量
            grp = GaussianRandomProjection(n_components=2000, random_state=42)
            vectors = grp.fit_transform(data_for_projection.T)

            # 基于余弦相似度构建图
            similarity_matrix = cosine_similarity(vectors)
            n_vectors = vectors.shape[0]
            edges = []
            for i in range(n_vectors):
                sim = similarity_matrix[i].copy()
                sim[i] = -np.inf  # 排除自身
                top8_indices = np.argsort(sim)[-8:]  # 取前8个最相似节点
                for j in top8_indices:
                    if i < j:  # 避免重复边
                        weight = similarity_matrix[i, j]
                        edges.append((i, j, float(weight)))

            # 构建无向图
            G = nx.Graph()
            G.add_nodes_from(range(n_vectors))
            for i, j, weight in edges:
                G.add_edge(i, j, weight=weight)

            fiber = G.copy()
            cortex = G.copy()

            # 节点值归一化（[0,1]）
            norm_value_fiber = (value_fiber - value_fiber.min()) / (value_fiber.max() - value_fiber.min())
            norm_value_cortex = (value_cortex - value_cortex.min()) / (value_cortex.max() - value_cortex.min())

            # 给每个节点构造 feature：harmonic_vec * normalized_value → 归一化
            for i in fiber.nodes():
                fiber.nodes[i]['original_value'] = float(value_fiber[i])
                fiber.nodes[i]['normalized_value'] = float(norm_value_fiber[i])
                harmonic_vec = vectors[i]
                features = harmonic_vec * norm_value_fiber[i]
                features_normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
                fiber.nodes[i]['features'] = features_normalized.tolist()

            for i in cortex.nodes():
                cortex.nodes[i]['original_value'] = float(value_cortex[i])
                cortex.nodes[i]['normalized_value'] = float(norm_value_cortex[i])
                harmonic_vec = vectors[i]
                features = harmonic_vec * norm_value_cortex[i]
                features_normalized = (features - features.min()) / (features.max() - features.min() + 1e-8)
                cortex.nodes[i]['features'] = features_normalized.tolist()

            # 再做全局归一化
            all_features_fiber = np.array([fiber.nodes[i]['features'] for i in fiber.nodes()])
            global_min_fiber = all_features_fiber.min(axis=0)
            global_max_fiber = all_features_fiber.max(axis=0)
            for i in fiber.nodes():
                feat = np.array(fiber.nodes[i]['features'])
                feat_global_norm = (feat - global_min_fiber) / (global_max_fiber - global_min_fiber + 1e-8)
                fiber.nodes[i]['features'] = feat_global_norm.tolist()

            all_features_cortex = np.array([cortex.nodes[i]['features'] for i in cortex.nodes()])
            global_min_cortex = all_features_cortex.min(axis=0)
            global_max_cortex = all_features_cortex.max(axis=0)
            for i in cortex.nodes():
                feat = np.array(cortex.nodes[i]['features'])
                feat_global_norm = (feat - global_min_cortex) / (global_max_cortex - global_min_cortex + 1e-8)
                cortex.nodes[i]['features'] = feat_global_norm.tolist()

            # 添加谱特征
            fiber = add_spectral_features(fiber, k=3)
            cortex = add_spectral_features(cortex, k=3)

            # GAT 注意力增强
            num_features = len(fiber.nodes[0]['features'])
            fiber = apply_gat_attention(fiber, num_features)
            cortex = apply_gat_attention(cortex, num_features)

            # 保存为 JSON 文件
            with open(os.path.join(output_folder, "fiber_normalized_sa.json"), "w") as f:
                json.dump(json_graph.node_link_data(fiber), f, indent=4, cls=NumpyEncoder)
            with open(os.path.join(output_folder, "cortex_normalized_sa.json"), "w") as f:
                json.dump(json_graph.node_link_data(cortex), f, indent=4, cls=NumpyEncoder)

            print(f"Saved files for {folder_name}")
        else:
            print(f"Skipping {folder_name}")
