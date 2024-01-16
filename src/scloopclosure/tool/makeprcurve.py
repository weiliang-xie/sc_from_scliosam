#! /usr/bin/python3
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import pandas as pd  
 
# 用Pandas读取csv格式的文件  
sc = pd.read_csv('../../../data/LOAMSCPRcurve/kitti_00.csv')
nd_a_c = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00_ca_filter_50.csv')
nd_c_a = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00_ca.csv')
nd_arccos = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00_ca_num.csv')

 
# 提取文件中的数据  
sc_x = sc['recall']  
sc_y = sc['presession']   
nd_a_c_x = nd_a_c['recall']
nd_a_c_y = nd_a_c['presession'] 
nd_c_a_x = nd_c_a['recall']
nd_c_a_y = nd_c_a['presession'] 
nd_arccos_x = nd_arccos['recall']
nd_arccos_y = nd_arccos['presession'] 
 
# 创建图像  
fig = plt.figure()  
 
# 绘制累计频率曲线  
plt.plot(sc_x,sc_y,'-r',linewidth = 1,label='SC') 
plt.plot(nd_a_c_x,nd_a_c_y,'-y',linewidth = 1,label='ND-ca-50')  
plt.plot(nd_c_a_x,nd_c_a_y,'-b',linewidth = 1,label='ND-ca')  
plt.plot(nd_arccos_x,nd_arccos_y,'-g',linewidth = 1,label='ND-ca_num')  
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