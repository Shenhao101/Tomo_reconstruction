import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import interp1d
import gc

#加载文件，k为文件的列数
def loadData(fname,fmin,fmax,ftype,k):
    data = []
    for num in range(fmin, fmax + 1):
        file = fname + str(num) + ftype
        f = open(file,'r')
        temp = []
        for each_line in f:
            temp1 = each_line.strip('\n')
            temp1 = temp1.split()
            if temp1 != []:
                temp.append(temp1)
        for i in range(len(temp)):
            for j in range(k):
                temp[i][j] = float(temp[i][j])
        data.append(temp)
        f.close()
    return data

#计算不同重建时间对应的深度
def get_depth():
    x = np.array([0, 8, 15, 23, 31, 38, 46, 54, 62, 69, 77, 85, 92, 100, 108, 115, 123, 131, 138,
                  146, 154, 162, 169, 177, 185, 192, 200, 208, 215, 223])
    y = np.arange(0,3000,100)
    f = interp1d(x, y, kind='cubic')
    depth = []
    for i in range(5, 221):#重建时间
        depth.append(f(i))
    return depth

def interpolation(data, interpdep):
    lenth = len(interpdep)
    value = []#储存插值后的数据
    for i in range (len(data[0])):
        depth = []
        dvp = []
        for j in range(len(data)):
            depth.append(6371 - data[j][i][0])
            dvp.append(data[j][i][2])
        depth = np.array(depth)
        max_dep = depth.max()#插值最大深度
        min_dep = depth.min()#插值最小深度
        dvp = np.array(dvp)
        f = interp1d(depth, dvp, kind = 'slinear')
        temp = []
        for j in range(lenth):#深度插值
            if interpdep[j] >= min_dep and interpdep[j] <= max_dep:
                temp.append(f(interpdep[j]))
            else:
                temp.append(0)
        value.append(temp)
    return value

#绘图
def plot_dvp(data, k, interpdep):#k表示层号
    v = []
    temp = []
    m = 0
    lenth = len(data)
    for i in range(lenth):
        temp.append(data[i][k])
        m = m + 1
        if m == 361 :
            m = 0
            v.append(temp)
            temp = []
    v = np.array(v)
    fig, ax = plt.subplots()
    #画等值线
    x = np.arange(-180, 181, 1)
    y = np.arange(-90, 91, 1)
    x, y = np.meshgrid(x, y)
    if interpdep[k] < 670:
        CS = ax.contour(x, y, v, levels=[0.5])
        #画层析成像模型
        im = ax.imshow(v, interpolation='bilinear', cmap=cm.jet_r, origin='lower',
                       extent=[-180, 180, -90, 90], vmax=2, vmin=-2)
    else:
        CS = ax.contour(x, y, v, levels=[0.25])
        im = ax.imshow(v, interpolation='bilinear', cmap=cm.jet_r, origin='lower',
                       extent=[-180, 180, -90, 90], vmax=1, vmin=-1)
    #去除图像周围白边
    plt.axis('off')
    height, width = v.shape
    fig.set_size_inches(width/30.0, height/30.0)
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)
    #存储图像
    name = 'LLNL_G3Dv3.' + str('%.1f'%interpdep[k]) + 'km-' + str(k+5) + '.png'
    fig.savefig(name, dpi=300)
    plt.close()
    gc.collect()
#main
data = loadData(fname='LLNL_G3Dv3.Interpolated.Layer', fmin=15, fmax=57, ftype='.txt', k=3)
print('data OK')
interpdep = get_depth()#要插值的深度值
dvp = interpolation(data, interpdep)
print('interpolation OK')
for k in range(len(dvp[0])):
    plot_dvp(dvp, k, interpdep)
print('plot OK')
