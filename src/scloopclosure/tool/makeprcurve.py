#! /usr/bin/python3
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import pandas as pd  
 
# 用Pandas读取csv格式的文件  
sc = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00_ca_height.csv')
a = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00_ca_filter_50.csv')
b = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00_ca_filter_100.csv')
c = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00_ca_num.csv')
d = pd.read_csv('../../../data/LOAMSCPRcurve/kitti_00_can-20.csv')


 
# 提取文件中的数据  
sc_x = sc['recall']  
sc_y = sc['presession']   
a_x = a['recall']
a_y = a['presession'] 
b_x = b['recall']
b_y = b['presession'] 
c_x = c['recall']
c_y = c['presession'] 
d_x = d['recall']
d_y = d['presession'] 
 
# 创建图像  
fig = plt.figure()  
 
# 绘制累计频率曲线  
plt.plot(sc_x,sc_y,'-r',linewidth = 1,label='ND_ca_height') 
plt.plot(a_x,a_y,'-y',linewidth = 1,label='ND_ca_fiter-50')  
plt.plot(b_x,b_y,'-b',linewidth = 1,label='ND_ca_filter-100')  
plt.plot(c_x,c_y,'-g',linewidth = 1,label='ND_ca_num')  
plt.plot(d_x,d_y,'-m',linewidth = 1,label='SC_can-20')  

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