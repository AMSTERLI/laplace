import numpy as np
import vtk

# 文件路径
vertices_file = 'full_brain_vertices.txt'
triangles_file = 'full_brain_triangles.txt'
stl_output_file = 'full_brain_with_labels.stl'

# 读取顶点数据，只取前三列 (x, y, z)
vertices = np.loadtxt(vertices_file, usecols=(0, 1, 2))

# 读取三角形数据，假设三角形文件中的每行是顶点索引 (整数)
# 并将索引从 1 开始的形式转换为 0 开始
triangles = np.loadtxt(triangles_file, dtype=int)

# 使用 VTK 构建并保存 STL 文件
def write_stl_with_vtk(filename, vertices, triangles):
    # 创建 vtkPoints 对象并插入顶点
    points = vtk.vtkPoints()
    for vert in vertices:
        points.InsertNextPoint(vert[0], vert[1], vert[2])

    # 创建 vtkCellArray 对象并插入三角形面片
    polys = vtk.vtkCellArray()
    for tri in triangles:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, tri[0])
        triangle.GetPointIds().SetId(1, tri[1])
        triangle.GetPointIds().SetId(2, tri[2])
        polys.InsertNextCell(triangle)

    # 创建 vtkPolyData 对象并添加顶点和三角形
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    # 创建 vtkSTLWriter 并设置输出文件名和数据
    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(filename)
    stl_writer.SetInputData(polydata)
    stl_writer.Write()

# 调用函数生成 STL 文件
write_stl_with_vtk(stl_output_file, vertices, triangles)
print(f"STL 文件已生成: {stl_output_file}")

# 读取白质表面 (左、右半球)
def load_white_surface(file):
    """加载白质表面文件，返回顶点"""
    with open(file, 'r') as f:
        f.readline()  # 跳过第一行
        n_verts = int(f.readline().split()[0])  # 获取顶点数
        verts = np.loadtxt(f, max_rows=n_verts, usecols=(0, 1, 2))  # 读取顶点数据，只取前三列 (x, y, z)
    return verts

# 加载左半球和右半球的白质表面
lh_white_verts = load_white_surface('/media/xcz/Upan/MRI-BRAIN/try-template/scriptsdata1/surface/lh.white.asc')
rh_white_verts = load_white_surface('/media/xcz/Upan/MRI-BRAIN/try-template/scriptsdata1/surface/rh.white.asc')

# 合并左右半球的白质顶点
white_verts = np.vstack([lh_white_verts, rh_white_verts])

# 读取 VTK 文件 (连接组)
vtk_reader = vtk.vtkPolyDataReader()
vtk_reader.SetFileName('connectome.vtk')
vtk_reader.Update()
vtk_polydata = vtk_reader.GetOutput()

# 获取连接组（VTK）和白质表面的中心
def get_center(polydata):
    bounds = polydata.GetBounds()
    center = [(bounds[i*2+1] + bounds[i*2]) / 2 for i in range(3)]
    return np.array(center)

white_center = np.mean(white_verts, axis=0)  # 白质表面的中心
vtk_center = get_center(vtk_polydata)  # 连接组的中心

# 平移 VTK 网格（连接组），使其与白质表面对齐
translation_vector = white_center - vtk_center
transform = vtk.vtkTransform()
transform.Translate(translation_vector)

# 将平移应用到 VTK 网格
transform_filter = vtk.vtkTransformPolyDataFilter()
transform_filter.SetTransform(transform)
transform_filter.SetInputData(vtk_polydata)
transform_filter.Update()
vtk_polydata_transformed = transform_filter.GetOutput()

# 保存对齐后的连接组为新的 VTK 文件
connectome_vtk_writer = vtk.vtkPolyDataWriter()
connectome_vtk_writer.SetFileName('aligned_connectome.vtk')
connectome_vtk_writer.SetInputData(vtk_polydata_transformed)
connectome_vtk_writer.Write()

# 保存对齐后的连接组为新的 STL 文件
connectome_stl_writer = vtk.vtkSTLWriter()
connectome_stl_writer.SetFileName('aligned_connectome.stl')
connectome_stl_writer.SetInputData(vtk_polydata_transformed)
connectome_stl_writer.Write()

print("已保存对齐后的连接组为 'aligned_connectome.vtk' 和 'aligned_connectome.stl'。")

# 保存皮质网格为 VTK 文件
# 创建一个 vtkPolyData 对象用于皮质网格
cortex_polydata = vtk.vtkPolyData()

# 创建顶点数据
points = vtk.vtkPoints()
for vert in vertices:
    points.InsertNextPoint(vert[0], vert[1], vert[2])

# 创建面数据
polys = vtk.vtkCellArray()
for tri in triangles:
    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, tri[0])
    triangle.GetPointIds().SetId(1, tri[1])
    triangle.GetPointIds().SetId(2, tri[2])
    polys.InsertNextCell(triangle)

# 将顶点和面添加到皮质网格中
cortex_polydata.SetPoints(points)
cortex_polydata.SetPolys(polys)

# 保存皮质网格为 VTK 文件
cortex_vtk_writer = vtk.vtkPolyDataWriter()
cortex_vtk_writer.SetFileName('cortex.vtk')
cortex_vtk_writer.SetInputData(cortex_polydata)
cortex_vtk_writer.Write()

# 保存皮质网格为 STL 文件
cortex_stl_writer = vtk.vtkSTLWriter()
cortex_stl_writer.SetFileName('cortex.stl')
cortex_stl_writer.SetInputData(cortex_polydata)
cortex_stl_writer.Write()

print("已保存皮质网格为 'cortex.vtk' 和 'cortex.stl'。")

# 合并对齐后的连接组和皮质网格
append_filter = vtk.vtkAppendPolyData()
append_filter.AddInputData(cortex_polydata)
append_filter.AddInputData(vtk_polydata_transformed)
append_filter.Update()
merged_polydata = append_filter.GetOutput()

# 保存合并后的网格为 VTK 文件
merged_vtk_writer = vtk.vtkPolyDataWriter()
merged_vtk_writer.SetFileName('merged_cortex_and_connectome.vtk')
merged_vtk_writer.SetInputData(merged_polydata)
merged_vtk_writer.Write()

# 保存合并后的网格为 STL 文件
merged_stl_writer = vtk.vtkSTLWriter()
merged_stl_writer.SetFileName('merged_cortex_and_connectome.stl')
merged_stl_writer.SetInputData(merged_polydata)
merged_stl_writer.Write()

print("已保存合并后的网格为 'merged_cortex_and_connectome.vtk' 和 'merged_cortex_and_connectome.stl'。")

# 拆分连接组
connectome_vtk_filename = 'aligned_connectome.vtk'

reader = vtk.vtkPolyDataReader()
reader.SetFileName(connectome_vtk_filename)
reader.Update()
polydata = reader.GetOutput()

# 创建顶点文件
points = polydata.GetPoints()
n_vertices = points.GetNumberOfPoints()
vertices = np.array([points.GetPoint(i) for i in range(n_vertices)])

# 创建向量（连接关系）文件
lines = polydata.GetLines()
lines.InitTraversal()
connections = []

idList = vtk.vtkIdList()
connection_index = 0
while lines.GetNextCell(idList):
    # Extract the entire line as a list of vertex indices
    connection = [idList.GetId(i) for i in range(idList.GetNumberOfIds())]
    connections.append(f"Connection {connection_index}: {connection}")
    connection_index += 1

# 保存顶点文件和向量文件
np.savetxt('connectome_vertices.txt', vertices)

with open('connectome_connections.txt', 'w') as f:
    for connection in connections:
        f.write(connection + '\n')
      
print("successful")

