import numpy as np
import vtk

# 假设 talairach_matrix.txt 中有4x4矩阵数据
transform_matrix = np.loadtxt('talairach_matrix.txt')

vtk_mat = vtk.vtkMatrix4x4()
for i in range(4):
    for j in range(4):
        vtk_mat.SetElement(i, j, transform_matrix[i, j])

transform = vtk.vtkTransform()
transform.SetMatrix(vtk_mat)

# 文件路径
vertices_file = 'full_brain_vertices.txt'
triangles_file = 'full_brain_triangles.txt'
stl_output_file = 'full_brain_with_labels.stl'

# 读取顶点数据 (x,y,z)
vertices = np.loadtxt(vertices_file, usecols=(0, 1, 2))
triangles = np.loadtxt(triangles_file, dtype=int)

def write_stl_with_vtk(filename, vertices, triangles):
    points = vtk.vtkPoints()
    for vert in vertices:
        points.InsertNextPoint(vert[0], vert[1], vert[2])

    polys = vtk.vtkCellArray()
    for tri in triangles:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, tri[0])
        triangle.GetPointIds().SetId(1, tri[1])
        triangle.GetPointIds().SetId(2, tri[2])
        polys.InsertNextCell(triangle)

    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(polys)

    stl_writer = vtk.vtkSTLWriter()
    stl_writer.SetFileName(filename)
    stl_writer.SetInputData(polydata)
    stl_writer.Write()

write_stl_with_vtk(stl_output_file, vertices, triangles)
print(f"STL 文件已生成: {stl_output_file}")

def load_white_surface(file):
    with open(file, 'r') as f:
        f.readline()  # 跳过第一行
        n_verts = int(f.readline().split()[0])
        verts = np.loadtxt(f, max_rows=n_verts, usecols=(0, 1, 2))
    return verts

lh_white_verts = load_white_surface('/media/xcz/Upan/MRI-BRAIN/try-template/scriptsdata1/surface/lh.white.asc')
rh_white_verts = load_white_surface('/media/xcz/Upan/MRI-BRAIN/try-template/scriptsdata1/surface/rh.white.asc')
white_verts = np.vstack([lh_white_verts, rh_white_verts])

# 读取 connectome.vtk（不变换）
vtk_reader = vtk.vtkPolyDataReader()
vtk_reader.SetFileName('connectome.vtk')
vtk_reader.Update()
vtk_polydata = vtk_reader.GetOutput()

# 构建 cortex polydata
cortex_polydata = vtk.vtkPolyData()
points = vtk.vtkPoints()
for vert in vertices:
    points.InsertNextPoint(vert[0], vert[1], vert[2])

polys = vtk.vtkCellArray()
for tri in triangles:
    triangle = vtk.vtkTriangle()
    triangle.GetPointIds().SetId(0, tri[0])
    triangle.GetPointIds().SetId(1, tri[1])
    triangle.GetPointIds().SetId(2, tri[2])
    polys.InsertNextCell(triangle)

cortex_polydata.SetPoints(points)
cortex_polydata.SetPolys(polys)

# 对 cortex_polydata 应用MNI空间变换
transform_filter = vtk.vtkTransformPolyDataFilter()
transform_filter.SetTransform(transform)
transform_filter.SetInputData(cortex_polydata)
transform_filter.Update()
cortex_polydata_mni = transform_filter.GetOutput()

# 保存应用变换后的 cortex 网格
cortex_vtk_writer = vtk.vtkPolyDataWriter()
cortex_vtk_writer.SetFileName('cortex_mni.vtk')
cortex_vtk_writer.SetInputData(cortex_polydata_mni)
cortex_vtk_writer.Write()

cortex_stl_writer = vtk.vtkSTLWriter()
cortex_stl_writer.SetFileName('cortex_mni.stl')
cortex_stl_writer.SetInputData(cortex_polydata_mni)
cortex_stl_writer.Write()

print("已保存应用变换后的皮质网格为 'cortex_mni.vtk' 和 'cortex_mni.stl'。")

# 合并时使用未变换的 connectome.vtk 数据
append_filter = vtk.vtkAppendPolyData()
append_filter.AddInputData(cortex_polydata_mni) # 已变换的 cortex
append_filter.AddInputData(vtk_polydata)        # 原始 connectome 不变
append_filter.Update()
merged_polydata = append_filter.GetOutput()

merged_vtk_writer = vtk.vtkPolyDataWriter()
merged_vtk_writer.SetFileName('merged_cortex_and_connectome_mni.vtk')
merged_vtk_writer.SetInputData(merged_polydata)
merged_vtk_writer.Write()

merged_stl_writer = vtk.vtkSTLWriter()
merged_stl_writer.SetFileName('merged_cortex_and_connectome_mni.stl')
merged_stl_writer.SetInputData(merged_polydata)
merged_stl_writer.Write()

print("已保存合并后的MNI空间网格为 'merged_cortex_and_connectome_mni.vtk' 和 'merged_cortex_and_connectome_mni.stl'。")

# 分析 connectome（原始）
connectome_vtk_filename = 'connectome.vtk'
reader = vtk.vtkPolyDataReader()
reader.SetFileName(connectome_vtk_filename)
reader.Update()
polydata = reader.GetOutput()

points = polydata.GetPoints()
n_vertices = points.GetNumberOfPoints()
vertices = np.array([points.GetPoint(i) for i in range(n_vertices)])

lines = polydata.GetLines()
lines.InitTraversal()
connections = []
idList = vtk.vtkIdList()
connection_index = 0
while lines.GetNextCell(idList):
    connection = [idList.GetId(i) for i in range(idList.GetNumberOfIds())]
    connections.append(f"Connection {connection_index}: {connection}")
    connection_index += 1

np.savetxt('connectome_vertices.txt', vertices)
with open('connectome_connections.txt', 'w') as f:
    for connection in connections:
        f.write(connection + '\n')

print("successful")
