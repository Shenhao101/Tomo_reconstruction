import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import matplotlib.cm as cm

#加载文件,k为文件的列数
def loadData(file, k):
    f = open(file, 'r')
    data = []
    for each_line in f:
        temp1 = each_line.strip('\n')
        temp2 = temp1.split()
        if temp2 != []:
            data.append(temp2)
    for i in range(len(data)):
        for j in range(k):
            data[i][j] = float(data[i][j])
    return data

#求平均深度
def get_AverDepth(data, k):
    depth = 0
    for i in range(len(data)):
        depth = depth + (6371 - data[i][k-1])
    Averdepth = depth / len(data)
    return Averdepth

#读取数据，k为要读取的列
def get_data(data, k):
    V = []
    temp = []
    m = 0
    for i in range(len(data)):
        temp.append(data[i][k-1])
        m = m + 1
        if m == 361 :
            m = 0
            V.append(temp)
            temp = []
    return V

#计算深度切片对应的年龄
def get_age(depth):
    x = np.arange(0,3000,100)
    y = np.array([0, 8, 15, 23, 31, 38, 46, 54, 62, 69, 77, 85, 92, 100, 108, 115, 123, 131, 138,
                  146, 154, 162, 169, 177, 185, 192, 200, 208, 215, 223])
    f = interp1d(x, y, kind='cubic')
    return float(f(depth))


data = loadData('LLNL_G3Dv3.Interpolated.Layer57_Lower_Mantle_2891km.txt', 3)
dVp = get_data(data, 3)
data_plot = np.array(dVp)
AverDepth = get_AverDepth(data,1)
age = get_age(AverDepth)
print(age)
#绘图
fig, ax = plt.subplots()
#画等值线
x = np.arange(-180,181,1)
y = np.arange(-90,91,1)
x, y = np.meshgrid(x,y)
CS = ax.contour(x, y, data_plot, levels=[1])

v = max(data_plot.max(), abs(data_plot.min()))
im = ax.imshow(data_plot, interpolation='bilinear', cmap=cm.jet_r,origin='lower',
               extent=[-180, 180, -90, 90], vmax=2, vmin=-2)

#去除图像周围白边
#height, width = data_plot.shape
#dpi = 600
#fig.set_size_inches(width/dpi, height/dpi) 

plt.axis('off')
plt.gca().xaxis.set_major_locator(plt.NullLocator()) 
plt.gca().yaxis.set_major_locator(plt.NullLocator()) 
plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0) 
plt.margins(0,0)
plt.show()

#fig.savefig('result.png', dpi=dpi)

