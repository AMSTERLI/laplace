#!/usr/bin/env python3
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg
import trimesh
import matplotlib.pyplot as plt
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
import ast
from scipy.io import savemat

# 定义计算两个向量之间角度的函数
def angle_between_vectors(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    # 防止数值误差
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return angle

# 1. 加载 STL 网格
mesh = trimesh.load('cortex.stl')

if not isinstance(mesh, trimesh.Trimesh):
    raise TypeError("Mesh is not a triangular mesh.")

vertices = mesh.vertices
faces = mesh.faces

n_vertices = len(vertices)

# 2. 计算顶点区域 A_i
triangle_areas = mesh.area_faces
vertex_areas = np.zeros(len(vertices))

for i, face in enumerate(faces):
    area = triangle_areas[i] / 3.0
    for vertex in face:
        vertex_areas[vertex] += area

# 保存顶点区域
np.save('vertex_areas.npy', vertex_areas)

# 3. 计算棱边的余切权重 w_{ij}
W = {}
epsilon_angle = 1e-8  # 防止数值问题

for face in faces:
    i, j, k = face
    vi = vertices[i]
    vj = vertices[j]
    vk = vertices[k]
    
    e_ij = vi - vj
    e_jk = vj - vk
    e_ki = vk - vi
    
    # 计算角度
    angle_i = angle_between_vectors(-e_ki, e_ij)
    angle_j = angle_between_vectors(-e_ij, e_jk)
    angle_k = angle_between_vectors(-e_jk, e_ki)
    
    # 防止角度为 0 或 π
    angle_i = np.clip(angle_i, epsilon_angle, np.pi - epsilon_angle)
    angle_j = np.clip(angle_j, epsilon_angle, np.pi - epsilon_angle)
    angle_k = np.clip(angle_k, epsilon_angle, np.pi - epsilon_angle)
    
    # 计算余切
    cot_i = 1.0 / np.tan(angle_i)
    cot_j = 1.0 / np.tan(angle_j)
    cot_k = 1.0 / np.tan(angle_k)
    
    for (m, n, cotangent) in [(i, j, cot_k), (j, k, cot_i), (k, i, cot_j)]:
        key = (min(m, n), max(m, n))
        W.setdefault(key, 0)
        W[key] += cotangent

# 初始化权重矩阵 W_matrix
from scipy.sparse import lil_matrix

W_matrix = lil_matrix((n_vertices, n_vertices))

for (i, j), w in W.items():
    W_matrix[i, j] = w
    W_matrix[j, i] = w  # 确保对称性

# 保存初始权重矩阵（在添加纤维束连接之前）
sp.save_npz('W_matrix_initial.npz', W_matrix.tocsr())

# 加载纤维束数据
# 加载纤维束顶点
connectome_vertices = np.loadtxt('connectome_vertices.txt')

# 加载纤维束连接信息
connections = []
with open('connectome_connections.txt', 'r') as f:
    for line in f:
        # 假设每行格式为："Connection X: [idx0, idx1, idx2, ...]"
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            indices_str = parts[1].strip()
            try:
                # 使用 ast.literal_eval 解析列表字符串
                indices = ast.literal_eval(indices_str)
                if isinstance(indices, list):
                    connections.append(indices)
                else:
                    print(f"Line does not contain a list: {line}")
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(e)
                continue  # 跳过错误的行

# 构建 KD 树
cortex_tree = cKDTree(vertices)

# 设置距离阈值，根据数据调整
distance_threshold = 0.05  # 根据需要调整

# 设置纤维束连接的权重
weight_fiber = 1.0  # 可以根据需要调整

# 遍历纤维束
for fiber_indices in connections:
    # 获取纤维束路径上的点坐标
    fiber_points = connectome_vertices[fiber_indices, :]
    
    # 对于纤维束上的每个点
    for point in fiber_points:
        # 在皮质网格中找到距离该点在阈值内的顶点索引
        nearby_indices = cortex_tree.query_ball_point(point, r=distance_threshold)
        # 在权重矩阵中添加连接
        for idx1 in nearby_indices:
            for idx2 in nearby_indices:
                if idx1 != idx2:
                    W_matrix[idx1, idx2] += weight_fiber
                    W_matrix[idx2, idx1] += weight_fiber  # 确保对称性

# 保存添加纤维束连接后的权重矩阵
sp.save_npz('W_matrix_with_fibers.npz', W_matrix.tocsr())

# 重新计算每个顶点的权重和
W_sum_new = np.array(W_matrix.sum(axis=1)).flatten()

# 重建拉普拉斯矩阵 L_new
row = []
col = []
data = []

W_coo = W_matrix.tocoo()
for i, j, w in zip(W_coo.row, W_coo.col, W_coo.data):
    if i != j:
        row.append(i)
        col.append(j)
        data.append(w)
    
for i in range(n_vertices):
    row.append(i)
    col.append(i)
    data.append(-W_sum_new[i])

L_new = sp.coo_matrix((data, (row, col)), shape=(n_vertices, n_vertices)).tocsr()

# 保存拉普拉斯矩阵
sp.save_npz('Laplacian_matrix.npz', L_new)

# 4. 构建质量矩阵 M
M = sp.diags(vertex_areas)

# 保存质量矩阵
sp.save_npz('Mass_matrix.npz', M)

# 5. 求解广义特征值问题 L_new φ = λ M φ

k = 100  # 计算前 100 个特征值和特征向量

# 求解广义特征值问题
eigenvalues, eigenvectors = eigsh(L_new, k=k, M=M, sigma=0, which='LM')

# 保存特征值和特征向量
np.save('eigenvalues.npy', eigenvalues)
np.save('eigenvectors.npy', eigenvectors)

# 或者保存为 MATLAB 文件
savemat('eigen_data.mat', {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors})

# 6. 可视化前几个特征函数
import matplotlib.cm as cm

for n in range(1, 10):  # 可视化前 10 个非平凡特征函数
    phi_n = eigenvectors[:, n]
    phi_n_normalized = (phi_n - np.min(phi_n)) / (np.max(phi_n) - np.min(phi_n))
    colors = cm.viridis(phi_n_normalized)
    mesh.visual.vertex_colors = (colors[:, :3] * 255).astype(np.uint8)
    mesh.show()
    
