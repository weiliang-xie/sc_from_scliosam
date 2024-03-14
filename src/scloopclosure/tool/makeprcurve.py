#! /usr/bin/python3
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import pandas as pd  

data_set_sq = "00"

sc_a_name = 'can-20'
sc_b_name = 'can-20_gt-small'
nd_a_name = 'ca_ve_transform_add-near-8-voxel_dist-2'
nd_b_name = 'ca_ve_transform_dist-2'
mix_a_name = 'ca_num_ve-test_can-20'
mix_b_name = 'ca_num_can-20'

 
# 用Pandas读取csv格式的文件  
sc_a = pd.read_csv('../../../data/LOAMSC/PRcurve/sc_kitti_' + data_set_sq + '_' + sc_a_name + '.csv')
# sc_b = pd.read_csv('../../../data/LOAMSC/PRcurve/sc_kitti_' + data_set_sq + '_' + sc_b_name + '.csv')
nd_a = pd.read_csv('../../../data/LOAMND/PRcurve/nd_kitti_' + data_set_sq + '_' + nd_a_name + '.csv')
nd_b = pd.read_csv('../../../data/LOAMND/PRcurve/nd_kitti_' + data_set_sq + '_' + nd_b_name + '.csv')
# mix_a = pd.read_csv('../../../data/LOAMMIX/PRcurve/mix_kitti_' + data_set_sq + '_' + mix_a_name + '.csv')
mix_b = pd.read_csv('../../../data/LOAMMIX/PRcurve/mix_kitti_' + data_set_sq + '_' + mix_b_name + '.csv')


 
# 提取文件中的数据  
sc_a_x = sc_a['recall']  
sc_a_y = sc_a['presession']   
# sc_b_x = sc_b['recall']  
# sc_b_y = sc_b['presession'] 
nd_a_x = nd_a['recall']
nd_a_y = nd_a['presession']   
nd_b_x = nd_b['recall']
nd_b_y = nd_b['presession'] 
# mix_a_x = mix_a['recall']
# mix_a_y = mix_a['presession'] 
mix_b_x = mix_b['recall']
mix_b_y = mix_b['presession'] 
 
# 创建图像  
fig = plt.figure()  
 
# 绘制累计频率曲线  
plt.plot(sc_a_x,sc_a_y,'-r',linewidth = 1,label='SC_' + sc_a_name) 
# plt.plot(sc_b_x,sc_b_y,'-m',linewidth = 1,label='SC_' + sc_b_name) 
plt.plot(nd_a_x,nd_a_y,'-b',linewidth = 1,label='ND_' + nd_a_name) 
plt.plot(nd_b_x,nd_b_y,'-c',linewidth = 1,label='ND_' + nd_b_name)  
# plt.plot(mix_a_x,mix_a_y,'-y',linewidth = 1,label='MIX_' + mix_a_name)  
plt.plot(mix_b_x,mix_b_y,'-g',linewidth = 1,label='MIX_' + mix_b_name)  

# plt.plot(x, BB, '-g')   # 绿色实线 
 
# 设置题目与坐标轴名称  
plt.title('SC PR Curve')  
plt.ylabel('presession')  
plt.xlabel('recall') 
 
# 设置图例（置于右下方）  
plt.legend(loc='lower left')  
# plt.legend(loc='medium')  

 
# 显示图片  
plt.show()