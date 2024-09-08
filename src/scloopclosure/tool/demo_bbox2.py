import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# 读取点云数据
pointcloud = o3d.io.read_point_cloud("/home/jtcx/remote_control/code/sc_from_scliosam/data/Scans_test/002348.pcd")

# 将点云转换为numpy数组
points = np.asarray(pointcloud.points)

# 根据Z值去除离群点
z_values = points[:, 2]  # 获取所有点的Z值
z_mean = np.mean(z_values)  # 计算Z值的均值
z_std = np.std(z_values)    # 计算Z值的标准差

# 定义离群点的过滤范围
z_threshold = 2.0  # 设置标准差的倍数
inlier_mask = (z_values > (z_mean - z_threshold * z_std)) & (z_values < (z_mean + z_threshold * z_std))
filtered_points = points[inlier_mask]  # 根据Z值过滤点

# 根据点的Z值设置颜色
filtered_z_values = filtered_points[:, 2]  # 获取过滤后的点的Z值
z_min, z_max = filtered_z_values.min(), filtered_z_values.max()

# 将高度标准化到0-1范围
normalized_z = (filtered_z_values - z_min) / (z_max - z_min)

# 使用matplotlib的colormap生成更深的蓝到红渐变色
colormap = cm.get_cmap('jet')  # 'jet' 是一个从深蓝到深红的渐变色带
colors = colormap(normalized_z)[:, :3]  # 获取 RGB 值，忽略 alpha 通道

# 创建可视化窗口
vis = o3d.visualization.Visualizer()
vis.create_window(window_name='pcd', width=1440, height=1080)

# 创建点云对象
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(filtered_points)
pcd.colors = o3d.utility.Vector3dVector(colors)

# 获取渲染选项并设置点大小
render_option = vis.get_render_option()
render_option.point_size = 2.0

# 添加点云到可视化
vis.add_geometry(pcd)

# 显示并运行可视化
vis.poll_events()
vis.update_renderer()
vis.run()

# 保存图像
vis.capture_screen_image("/home/jtcx/ICRA/exper_data_1.0/pr_eps/point_cloud/pt_image.png")
vis.destroy_window()
