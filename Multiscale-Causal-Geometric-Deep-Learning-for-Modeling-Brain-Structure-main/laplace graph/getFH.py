import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree
import ast
from scipy.io import savemat
from scipy.sparse.linalg import eigsh
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 定义计算两个向量之间角度的函数
def angle_between_vectors(u, v):
    cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.arccos(cos_theta)

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
    vi, vj, vk = vertices[i], vertices[j], vertices[k]
    e_ij, e_jk, e_ki = vi - vj, vj - vk, vk - vi
    angle_i = angle_between_vectors(-e_ki, e_ij)
    angle_j = angle_between_vectors(-e_ij, e_jk)
    angle_k = angle_between_vectors(-e_jk, e_ki)
    angle_i = np.clip(angle_i, epsilon_angle, np.pi - epsilon_angle)
    angle_j = np.clip(angle_j, epsilon_angle, np.pi - epsilon_angle)
    angle_k = np.clip(angle_k, epsilon_angle, np.pi - epsilon_angle)
    cot_i, cot_j, cot_k = 1.0 / np.tan(angle_i), 1.0 / np.tan(angle_j), 1.0 / np.tan(angle_k)
    for (m, n, cotangent) in [(i, j, cot_i), (j, k, cot_j), (k, i, cot_k)]:
        key = (min(m, n), max(m, n))
        W.setdefault(key, 0)
        W[key] += cotangent

from scipy.sparse import lil_matrix
W_matrix = lil_matrix((n_vertices, n_vertices))

for (i, j), w in W.items():
    W_matrix[i, j] = w
    W_matrix[j, i] = w  # 确保对称性

# 将 W_matrix 转换为 CSR 格式，然后计算最大值
W_matrix = W_matrix.tocsr()
max_weight = W_matrix.max()
if max_weight > 0:
    W_matrix /= max_weight

sp.save_npz('W_matrix_initial_normalized.npz', W_matrix)

# 4. 加载纤维束数据
connectome_vertices = np.loadtxt('connectome_vertices.txt')

connections = []
with open('connectome_connections.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            indices_str = parts[1].strip()
            indices = ast.literal_eval(indices_str)
            if isinstance(indices, list):
                connections.append(indices)

# 加载每条纤维的平均FA值并归一化
fiber_FA = {}
fa_values = []

with open('fibers_mean_FA.txt', 'r') as fa_file:
    for line in fa_file:
        index, fa_value = line.strip().split()
        fa_value = float(fa_value)
        fiber_FA[int(index)] = fa_value
        fa_values.append(fa_value)

# 对FA值进行归一化
fa_min, fa_max = np.min(fa_values), np.max(fa_values)
for key in fiber_FA:
    fiber_FA[key] = (fiber_FA[key] - fa_min) / (fa_max - fa_min)

cortex_tree = cKDTree(vertices)
distance_threshold = 0.05

# 添加一个变量来控制是否使用FA更新权重
use_FA_weights = False  # 如果是皮质表面三角形，则设置为 False

if use_FA_weights:
    # 根据归一化后的FA值更新权重矩阵
    for fiber_index, fiber_indices in enumerate(connections):
        fa_weight = fiber_FA.get(fiber_index + 1, 0)
        fiber_points = connectome_vertices[fiber_indices, :]
        for point in fiber_points:
            nearby_indices = cortex_tree.query_ball_point(point, r=distance_threshold)
            for idx1 in nearby_indices:
                for idx2 in nearby_indices:
                    if idx1 != idx2:
                        W_matrix[idx1, idx2] += fa_weight
                        W_matrix[idx2, idx1] += fa_weight

sp.save_npz('W_matrix_with_FA_normalized.npz', W_matrix)

# 重新计算拉普拉斯矩阵
W_sum_new = np.array(W_matrix.sum(axis=1)).flatten()
row, col, data = [], [], []
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
sp.save_npz('Laplacian_matrix.npz', L_new)

# 构建质量矩阵并求解特征值问题
M = sp.diags(vertex_areas)
sp.save_npz('Mass_matrix.npz', M)
k = 100
eigenvalues, eigenvectors = eigsh(L_new, k=k, M=M, sigma=0, which='LM')
np.save('eigenvalues.npy', eigenvalues)
np.save('eigenvectors.npy', eigenvectors)
savemat('eigen_data.mat', {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors})

# 可视化前几个特征函数
for n in range(1, 10):
    phi_n = eigenvectors[:, n]
    phi_n_normalized = (phi_n - np.min(phi_n)) / (np.max(phi_n) - np.min(phi_n))
    colors = cm.viridis(phi_n_normalized)
    mesh.visual.vertex_colors = (colors[:, :3] * 255).astype(np.uint8)
    mesh.show()
