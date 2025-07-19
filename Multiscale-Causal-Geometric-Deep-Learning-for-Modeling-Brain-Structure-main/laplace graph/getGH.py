import numpy as np
import vtk
import ast
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from scipy.io import savemat
from vtk.util import numpy_support
import trimesh
import matplotlib.cm as cm

def load_cortex_mesh(stl_file):
    reader = vtk.vtkSTLReader()
    reader.SetFileName(stl_file)
    reader.Update()
    polydata = reader.GetOutput()
    
    # 提取顶点坐标
    points = polydata.GetPoints()
    vtk_array = points.GetData()
    cortex_vertices = numpy_support.vtk_to_numpy(vtk_array)
    
    # 提取三角形面信息
    polys = polydata.GetPolys().GetData()
    polys_array = numpy_support.vtk_to_numpy(polys)
    cortex_triangles = polys_array.reshape(-1, 4)[:, 1:4]  # 跳过第一个计数值
    
    return cortex_vertices, cortex_triangles.astype(np.int64)

# 加载皮质网格
cortex_vertices, cortex_triangles = load_cortex_mesh('cortex.stl')
n_cortex_vertices = cortex_vertices.shape[0]

# 加载纤维束顶点
connectome_vertices = np.loadtxt('connectome_vertices.txt')

# 加载纤维束连接信息
connections = []
with open('connectome_connections.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(':', 1)
        if len(parts) == 2:
            indices_str = parts[1].strip()
            try:
                indices = ast.literal_eval(indices_str)
                if isinstance(indices, list):
                    connections.append(indices)
                else:
                    print(f"Line does not contain a list: {line}")
            except Exception as e:
                print(f"Error parsing line: {line}")
                print(e)
                continue

# 构建 KD 树
cortex_tree = cKDTree(cortex_vertices)

# **优化：使用边列表构建连接矩阵**
# 提取皮质网格的拓扑连接关系
edges_topology = np.vstack([
    cortex_triangles[:, [0, 1]],
    cortex_triangles[:, [1, 2]],
    cortex_triangles[:, [2, 0]]
])

# 确保对称性
edges_topology = np.vstack([edges_topology, edges_topology[:, [1, 0]]])

# 初始化纤维束的边列表
edges_fibers = []

# 设置距离阈值和插值点数量
distance_threshold = 0.05  # 根据需要调整
num_interp_points = 0  # 根据需要调整

# 遍历纤维束
for fiber_indices in connections:
    fiber_points = connectome_vertices[fiber_indices, :]
    for i in range(len(fiber_points) - 1):
        start_point = fiber_points[i]
        end_point = fiber_points[i + 1]
        
        # 插值
        if num_interp_points > 0:
            t_values = np.linspace(0, 1, num_interp_points)
            interpolated_points = np.outer(1 - t_values, start_point) + np.outer(t_values, end_point)
        else:
            interpolated_points = [start_point, end_point]
        
        # 在插值点上寻找靠近的皮质顶点
        for point in interpolated_points:
            nearby_indices = cortex_tree.query_ball_point(point, r=distance_threshold)
            indices = np.array(nearby_indices)
            if len(indices) >= 2:
                idx1, idx2 = np.meshgrid(indices, indices)
                mask = idx1 != idx2  # 去除自环
                idx_pairs = np.vstack([idx1[mask], idx2[mask]]).T
                edges_fibers.append(idx_pairs)
            elif len(indices) == 1:
                pass  # 只有一个点，无法形成边

# 将所有纤维束的边合并
if edges_fibers:
    edges_fibers = np.vstack(edges_fibers)
    # **确保对称性**
    edges_fibers = np.vstack([edges_fibers, edges_fibers[:, [1, 0]]])
else:
    edges_fibers = np.empty((0, 2), dtype=int)

# 合并所有边
edges_all = np.vstack([edges_topology, edges_fibers])

# 移除重复边
edges_all_sorted = np.sort(edges_all, axis=1)
edges_unique = np.unique(edges_all_sorted, axis=0)

# 构建稀疏连接矩阵
adjacency = sp.csr_matrix(
    (np.ones(edges_unique.shape[0]), (edges_unique[:, 0], edges_unique[:, 1])),
    shape=(n_cortex_vertices, n_cortex_vertices)
)

# **确保邻接矩阵是对称的**
adjacency = adjacency.maximum(adjacency.T)

# 计算度矩阵 D
degree = adjacency.sum(axis=1).A1
degree_matrix = sp.diags(degree)

# 计算对称拉普拉斯矩阵
laplacian = degree_matrix - adjacency

# 检查拉普拉斯矩阵的对称性
if (laplacian - laplacian.T).nnz == 0:
    print("拉普拉斯矩阵是对称的")
else:
    print("拉普拉斯矩阵不是对称的")

# 计算特征值和特征向量
num_eigenvalues = 10  # 根据需要调整
eigenvalues, eigenvectors = eigsh(laplacian, k=num_eigenvalues, which='SM')

# 保存特征值和特征向量
savemat('harmonics.mat', {'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors})

# 保存连接矩阵
savemat('updated_adjacency_matrix.mat', {'adjacency': adjacency})
print("计算完成并已保存结果。")

# **可视化前几个特征函数**
# 创建 trimesh 网格对象
mesh = trimesh.Trimesh(vertices=cortex_vertices, faces=cortex_triangles)

for n in range(1, 10):  # 可视化前 10 个非平凡特征函数
    phi_n = eigenvectors[:, n]
    phi_n_normalized = (phi_n - np.min(phi_n)) / (np.max(phi_n) - np.min(phi_n))
    colors = cm.viridis(phi_n_normalized)
    mesh.visual.vertex_colors = (colors[:, :3] * 255).astype(np.uint8)
    mesh.show()
    
   
