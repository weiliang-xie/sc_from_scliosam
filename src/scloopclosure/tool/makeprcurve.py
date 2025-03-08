#! /usr/bin/python3
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import pandas as pd  

data_set_sq_1 = "00"
data_set_sq_2 = "02"
data_set_sq_3 = "05"
data_set_sq_4 = "08"
data_set_sq_5 = "k1"

sc_a_name = 'base'
sc_b_name = 'base'
base_sq_a_name = 'base'
base_sq_b_name = 'base'
nd_a_name = 'avg-z'
nd_b_name = 'avg-desc'
mix_a_name = 'ca_num_ve-test_can-20'
mix_b_name = 'ca-num'

 
# 用Pandas读取csv格式的文件  
sc_a = pd.read_csv('/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMSC/PRcurve/sc_kitti_00_untitled.csv')
sc_b = pd.read_csv('/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMSC/PRcurve/sc_kitti_02_untitled.csv')
base_a = pd.read_csv('/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMSC/PRcurve/sc_kitti_05_untitled.csv')
base_b = pd.read_csv('/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMSC/PRcurve/sc_kitti_08_untitled.csv')
nd_a = pd.read_csv('/home/jtcx/remote_control/code/sc_from_scliosam/data/LOAMSC/PRcurve/sc_kitti_kaist01_untitled.csv')
# nd_b = pd.read_csv('../../../data/LOAMND/PRcurve/nd_kitti_' + data_set_sq_1 + '_' + nd_b_name + '.csv')
# mix_a = pd.read_csv('../../../data/LOAMMIX/PRcurve/mix_kitti_' + data_set_sq_1 + '_' + mix_a_name + '.csv')
# mix_b = pd.read_csv('../../../data/LOAMMIX/PRcurve/mix_kitti_' + data_set_sq_1 + '_' + mix_b_name + '.csv')


 
# 提取文件中的数据  
sc_a_x = sc_a['recall']  
sc_a_y = sc_a['precision']   
sc_b_x = sc_b['recall']  
sc_b_y = sc_b['precision'] 
base_a_x = base_a['recall']  
base_a_y = base_a['precision']   
base_b_x = base_b['recall']  
base_b_y = base_b['precision']
nd_a_x = nd_a['recall']
nd_a_y = nd_a['precision']   
# nd_b_x = nd_b['recall']
# nd_b_y = nd_b['precision'] 
# mix_a_x = mix_a['recall']
# mix_a_y = mix_a['precision'] 
# mix_b_x = mix_b['recall']
# mix_b_y = mix_b['precision'] 
 
# 创建图像  
fig = plt.figure()  
 
# 绘制累计频率曲线  
plt.plot(sc_a_x,sc_a_y,'-r',linewidth = 1,label='SC_' + data_set_sq_1 + '_' + sc_a_name) 
plt.plot(sc_b_x,sc_b_y,'-m',linewidth = 1,label='SC_' + data_set_sq_2 + '_' + sc_b_name) 
plt.plot(base_a_x,base_a_y,'-b',linewidth = 1,label='SC_' + data_set_sq_3 + '_' + sc_b_name) 
plt.plot(base_b_x,base_b_y,'-c',linewidth = 1,label='SC_' + data_set_sq_4 + '_' + sc_b_name) 
plt.plot(nd_a_x,nd_a_y,'-y',linewidth = 1,label='SC_' + data_set_sq_5 + '_' + sc_b_name) 
# plt.plot(nd_b_x,nd_b_y,'-c',linewidth = 1,label='ND_' + data_set_sq_1 + '_' + nd_b_name)  
# plt.plot(mix_a_x,mix_a_y,'-y',linewidth = 1,label='MIX_' + data_set_sq_1 + '_' + mix_a_name)  
# plt.plot(mix_b_x,mix_b_y,'-g',linewidth = 1,label='MIX_' + data_set_sq_1 + '_' + mix_b_name)  

# plt.plot(x, BB, '-g')   # 绿色实线 
 
# 设置题目与坐标轴名称  
plt.title('PR Curve')  
plt.ylabel('precision')  
plt.xlabel('recall') 
 
# 设置图例（置于右下方）  
plt.legend(loc='lower left')  
# plt.legend(loc='medium')  

 
# 显示图片  
plt.show()