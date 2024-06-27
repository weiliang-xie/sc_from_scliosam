import open3d as o3d
import numpy as np

def create_cylinder_outline(radius, min_height, max_height, ring,resolution=20):
    """
    创建圆柱体的轮廓线

    Parameters:
    - radius: 圆柱体的半径
    - height: 圆柱体的高度
    - resolution: 圆柱体的分辨率

    Returns:
    - lineset: 圆柱体的轮廓线对象
    """

    # 创建圆柱体的顶点
    vertices = []
    for i in range(1,ring + 1):
        for j in range(resolution):
            angle = 2 * np.pi * j / resolution
            x = i * (radius / ring) * np.cos(angle)
            y = i * (radius / ring) * np.sin(angle)
            vertices.append([x, y, min_height])
            vertices.append([x, y, max_height])

    # 创建圆柱体的轮廓线（边缘）
    lines = []
    for j in range(ring):
        for i in range(resolution):
            # 侧面边缘线
            lines.append([2 * i + j * resolution * 2, (2 * i + 2) % (2 * resolution) + j * resolution * 2])
            lines.append([2 * i + 1 + j * resolution * 2, (2 * i + 3) % (2 * resolution) + j * resolution * 2])

    # 添加圆柱体顶部和底部的顶点
    vertices.append([0, 0, min_height])
    vertices.append([0, 0, max_height])

    # 顶部和底部的辐射线
    lines.extend([[2 * i + 2 * resolution * (ring - 1), 2 * resolution * ring] for i in range(resolution)])
    lines.extend([[2 * i + 1 + 2 * resolution * (ring - 1), 2 * resolution * ring + 1] for i in range(resolution)])

   # 添加中心垂直线
    lines.append([2 * resolution * ring, 2 * resolution * ring + 1])  # 顶底中心线

    # 添加圆环侧面垂直线
    for j in range(ring):
        for i in range(resolution):
            lines.append([2 * i + 2 * resolution * j, 2 * i + 1 + 2 * resolution * j])  # 侧面顶底连接线



    # 创建 LineSet 对象
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(vertices)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    color = [[0.529, 0.808, 0.922] for i in range(len(lines))]
    lineset.colors = o3d.utility.Vector3dVector(color)

    return lineset


pointcloud = o3d.io.read_point_cloud("/home/jtcx/remote_control/code/sc_from_scliosam/data/Scans_test/000000.pcd")
points = np.asarray(pointcloud.points)
pointcloud_cp = o3d.io.read_point_cloud("/home/jtcx/remote_control/code/sc_from_scliosam/data/Scans_test/004438.pcd")
points_cp = np.asarray(pointcloud_cp.points)

vis = o3d.visualization.Visualizer()
# vis_cp = o3d.visualization.Visualizer()
vis.create_window(window_name='pcd', width=1440, height=1080) #, width=800, height=600      #创建窗口
# vis_cp.create_window(window_name='pcd_cp', width=1440, height=1080) #, width=800, height=600      #创建窗口
render_option: o3d.visualization.RenderOption = vis.get_render_option() #get_render_option() 获取渲染对象句柄 “：” 表示类型注解，说明render_option的类型是o3d.visualization.RenderOption
# render_option_cp: o3d.visualization.RenderOption = vis_cp.get_render_option() #get_render_option() 获取渲染对象句柄 “：” 表示类型注解，说明render_option的类型是o3d.visualization.RenderOption
render_option.point_size = 2.0      #控制点云的点的大小
# render_option_cp.point_size = 2.0      #控制点云的点的大小

pcd = o3d.geometry.PointCloud()
pcd_cp = o3d.geometry.PointCloud()

pcd.points = o3d.utility.Vector3dVector(points)     #numpy转open3d
pcd_cp.points = o3d.utility.Vector3dVector(points_cp)     #numpy转open3d_cp

pcd.paint_uniform_color([0, 0, 0])      #点云显示为黑色
pcd_cp.paint_uniform_color([0, 0, 255])      #点云显示为黑色

aabb = pcd.get_axis_aligned_bounding_box()      #获取轴对称边界框


#建立圆环体素结构
cylinder_outline = create_cylinder_outline(radius = 80, min_height = -4, max_height = aabb.get_max_bound()[2], ring = 16, resolution = 40)

vis.add_geometry(cylinder_outline)
vis.add_geometry(pcd)           #添加点云
vis.add_geometry(pcd_cp)           #添加点云
# vis_cp.add_geometry(cylinder_outline)
# vis_cp.add_geometry(pcd_cp)           #添加点云


vis.poll_events()
vis.update_renderer()
vis.run()  # user changes the view and press "q" to terminate
param = vis.get_view_control().convert_to_pinhole_camera_parameters()
vis.destroy_window()

# vis_cp.poll_events()
# vis_cp.update_renderer()
# vis_cp.run()  # user changes the view and press "q" to terminate
# param = vis_cp.get_view_control().convert_to_pinhole_camera_parameters()
# vis_cp.destroy_window()


