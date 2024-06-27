import numpy as np
import open3d as o3d
# from helper import read_ply

pointcloud = o3d.io.read_point_cloud("/home/jtcx/remote_control/code/sc_from_scliosam/data/Scans_test/000000.pcd")
points = np.asarray(pointcloud.points)
# data = read_ply("F:\\visualization_SemanticKITTI\\data_kitti\\mini\\gt_1000\\000000ply.ply")
# points = np.vstack((data['x'], data['y'], data['z'])).T

# points = np.loadtxt("F:\\visualization_SemanticKITTI\\bbox.txt")

vis = o3d.visualization.Visualizer()
vis.create_window(window_name='pcd', width=1440, height=1080) #, width=800, height=600      #创建窗口
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(points)     #numpy转open3d
render_option: o3d.visualization.RenderOption = vis.get_render_option() #get_render_option() 获取渲染对象句柄 “：” 表示类型注解，说明render_option的类型是o3d.visualization.RenderOption
render_option.point_size = 2.0      #控制点云的点的大小

pcd.paint_uniform_color([0, 0, 0])      #点云显示为黑色

aabb = pcd.get_axis_aligned_bounding_box()      #获取轴对称边界框
aabb.color = (1, 0, 0)  # 设置颜色为红色

# 指定体素大小
voxel_size = [10, 10, 10]

# 计算每个子块的大小
num_divisions = np.ceil((aabb.get_max_bound() - aabb.get_min_bound()) / np.array(voxel_size)).astype(int)   #astype 转换数组数据类型

blocks = []
for i in range(num_divisions[0]):
    for j in range(num_divisions[1]):
        for k in range(num_divisions[2]):
            # 计算子块的边界框
            block_min_bound = aabb.get_min_bound() + np.array(voxel_size) * np.array([i, j, k])
            block_max_bound = block_min_bound + np.array(voxel_size)
            block_aabb = o3d.geometry.AxisAlignedBoundingBox(block_min_bound, block_max_bound)  #定义boundingbox
            block_aabb.color = (0.529, 0.808, 0.922)  # 设置颜色为绿色
            blocks.append(block_aabb)       #向列表添加boundingbox元素


vis.add_geometry(pcd)           #添加点云
for block in blocks:
    vis.add_geometry(block)     #添加boundingbox


vis.poll_events()
vis.update_renderer()
vis.run()  # user changes the view and press "q" to terminate
param = vis.get_view_control().convert_to_pinhole_camera_parameters()
vis.destroy_window()


# #圆环体素demo
# import open3d as o3d
# import numpy as np

# def create_cylinder_outline(radius, height, ring,resolution=20):
#     """
#     创建圆柱体的轮廓线

#     Parameters:
#     - radius: 圆柱体的半径
#     - height: 圆柱体的高度
#     - resolution: 圆柱体的分辨率

#     Returns:
#     - lineset: 圆柱体的轮廓线对象
#     """

#     # 创建圆柱体的顶点
#     vertices = []
#     for i in range(1,ring + 1):
#         for j in range(resolution):
#             angle = 2 * np.pi * j / resolution
#             x = i * (radius / ring) * np.cos(angle)
#             y = i * (radius / ring) * np.sin(angle)
#             vertices.append([x, y, 0])
#             vertices.append([x, y, height])

#     # 创建圆柱体的轮廓线（边缘）
#     lines = []
#     for j in range(ring):
#         for i in range(resolution):
#             # 侧面边缘线
#             lines.append([2 * i + j * resolution * 2, (2 * i + 2) % (2 * resolution) + j * resolution * 2])
#             lines.append([2 * i + 1 + j * resolution * 2, (2 * i + 3) % (2 * resolution) + j * resolution * 2])

#     # 添加圆柱体顶部和底部的顶点
#     vertices.append([0, 0, 0])
#     vertices.append([0, 0, height])

#     # 顶部和底部的辐射线
#     lines.extend([[2 * i + 2 * resolution * (ring - 1), 2 * resolution * ring] for i in range(resolution)])
#     lines.extend([[2 * i + 1 + 2 * resolution * (ring - 1), 2 * resolution * ring + 1] for i in range(resolution)])

#    # 添加中心垂直线
#     lines.append([2 * resolution * ring, 2 * resolution * ring + 1])  # 顶底中心线

#     # 添加圆环侧面垂直线
#     for j in range(ring):
#         for i in range(resolution):
#             lines.append([2 * i + 2 * resolution * j, 2 * i + 1 + 2 * resolution * j])  # 侧面顶底连接线



#     # 创建 LineSet 对象
#     lineset = o3d.geometry.LineSet()
#     lineset.points = o3d.utility.Vector3dVector(vertices)
#     lineset.lines = o3d.utility.Vector2iVector(lines)

#     return lineset

# # 创建一个半径为1，高度为2的圆柱体的轮廓线
# cylinder_outline = create_cylinder_outline(radius=3, height=2, ring = 3, resolution=40)

# # 创建可视化窗口并添加圆柱体的轮廓线
# vis = o3d.visualization.Visualizer()
# vis.create_window()
# vis.add_geometry(cylinder_outline)

# # 设置相机参数和视角
# ctr = vis.get_view_control()
# ctr.set_lookat([0, 0, 0])
# ctr.set_front([1, 0, 0])
# ctr.set_up([0, 0, 1])
# ctr.set_zoom(0.8)

# # 运行可视化窗口
# vis.run()
# vis.destroy_window()