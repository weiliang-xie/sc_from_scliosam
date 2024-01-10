#! /usr/bin/python3
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import pandas as pd  
 
# 用Pandas读取csv格式的文件  
sc = pd.read_csv('../../../data/LOAMSCPRcurve/kitti_00.csv')
nd = pd.read_csv('../../../data/LOAMNDPRcurve/kitti_00.csv')

 
# 提取文件中的数据  
sc_x = sc['recall']  
sc_y = sc['presession']   
nd_x = nd['recall']
nd_y = nd['presession'] 
 
# 创建图像  
fig = plt.figure()  
 
# 绘制累计频率曲线  
plt.plot(sc_x,sc_y,'-g',linewidth = 1,label='SC') 
plt.plot(nd_x,nd_y,'-y',linewidth = 1,label='ND')  
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