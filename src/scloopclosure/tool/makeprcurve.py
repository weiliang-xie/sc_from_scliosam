#! /usr/bin/python3
import matplotlib as mpl  
import matplotlib.pyplot as plt  
import pandas as pd  
 
# 用Pandas读取csv格式的文件  
sj = pd.read_csv('../../../data/LOAMPRcurve/kitti_00.csv')
 
# 提取文件中的数据  
x = sj['recall']  
BB = sj['presession']   
 
# 创建图像  
fig = plt.figure()  
 
# 绘制累计频率曲线  
plt.plot(x,BB,'-g',linewidth = 1)  
# plt.plot(x, BB, '-g')   # 绿色实线 
 
# 设置题目与坐标轴名称  
plt.title('SC PR Curve')  
plt.ylabel('presession')  
plt.xlabel('recall') 
 
# 设置图例（置于右下方）  
# plt.legend(loc='lower right')  
# plt.legend(loc='medium')  

 
# 显示图片  
plt.show()