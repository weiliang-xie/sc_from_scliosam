import matplotlib.pyplot as plt
import numpy as np

#python工具 像素显示sc描述符
def file2array(path, delimiter=' '):     # delimiter是数据分隔符
    fp = open(path, 'r', encoding='utf-8')
    string = fp.read()              # string是一行字符串，该字符串包含文件所有内容
    fp.close()
    row_list = string.splitlines()  # splitlines默认参数是‘\n’
    data_list = [[float(i) for i in row.strip().split(delimiter)] for row in row_list]
    return np.array(data_list)
 
data = file2array('../../../data/LOAMSCDs/000022.scd')
print(data)
print("data's shape", data.shape)


print(data)

fig = plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
plt.imshow(data, cmap='jet', interpolation='nearest')
plt.colorbar(shrink=0.3)
plt.show()

