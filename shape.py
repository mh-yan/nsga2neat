import math
import pickle
import data2txt
import neat2
import  numpy as np
from matplotlib import pyplot as plt
from matplotlib.tri import Triangulation
import utils

def triangulation(shapex,shapey):
    num_squares_x = int((shapex - 1) / 2)
    num_squares_y = int((shapey - 1) / 2)
    num_squares = int(num_squares_x * num_squares_y)
    Tri = np.zeros((num_squares * 8, 3))
    Index = np.zeros((shapey, shapex))
    # n从1开始
    n=0
    k = 0
    for i in range(shapey):
        for j in range(shapex):
            Index[i, j] = n
            n += 1
    for ii in range(num_squares_y):
        for jj in range(num_squares_x):
            # i,j is the index of point which is the left top of the square
            i = ii * 2
            j = jj * 2

            # ====================画三角形
            Tri[k, :] = [Index[i, j], Index[i + 1, j], Index[i, j + 1]]
            Tri[k + 1, :] = [Index[i + 1, j], Index[i + 1, j + 1], Index[i, j + 1]]
            Tri[k + 2, :] = [Index[i, j + 1], Index[i + 1, j + 1], Index[i + 1, j + 2]]
            Tri[k + 3, :] = [Index[i, j + 2], Index[i, j + 1], Index[i + 1, j + 2]]
            Tri[k + 4, :] = [Index[i + 1, j], Index[i + 2, j], Index[i + 2, j + 1]]
            Tri[k + 5, :] = [Index[i + 1, j], Index[i + 2, j + 1], Index[i + 1, j + 1]]
            Tri[k + 6, :] = [Index[i + 1, j + 1], Index[i + 2, j + 1], Index[i + 1, j + 2]]
            Tri[k + 7, :] = [Index[i + 2, j + 1], Index[i + 2, j + 2], Index[i + 1, j + 2]]
            k += 8
    return Tri

def find_contour(a, thresh, pcd):
    num_squares_x = int((a.shape[1] - 1) / 2)
    num_squares_y = int((a.shape[0] - 1) / 2)
    num_squares = int(num_squares_x * num_squares_y)
    X = pcd[:, 0].reshape(a.shape[0], a.shape[1])  # 先横向再纵向
    Y = pcd[:, 1].reshape(a.shape[0], a.shape[1])
    if np.isnan(X).any():
        print("you x")
    if np.isnan(Y).any():
        print("you y")
    Index = np.zeros((a.shape[0], a.shape[1]))
    Cat = (a > thresh) + 1 # 0,1,2  1--out 2--in  0--border
    # create index 先横向再纵向
    n = 0  # 点
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            Index[i, j] = n
            n += 1

    new_a = np.copy(a)
    new_a[a > thresh] = 1
    new_a[a < thresh] = -1
    l = new_a[:, 0:-2]
    r = new_a[:, 2:]
    t = new_a[0:-2, :]
    b = new_a[2:, :]
    flag_x = t * b
    flag_y = l * r
    k = 0
    a = a - thresh
    min_r = 0.3
    max_r = 0.7
    # ii,jj is the index of square
    for ii in range(num_squares_y):
        for jj in range(num_squares_x):

            # i,j is the index of point which is the left top of the square
            i = ii * 2
            j = jj * 2
            if Cat[i,j]==Cat[i,j+2]:
                Cat[i,j+1]=Cat[i,j]
            if Cat[i+2,j]==Cat[i+2,j+2]:
                Cat[i+2,j+1]=Cat[i+2,j+2]
            if Cat[i,j]==Cat[i+2,j]:
                Cat[i+1,j]=Cat[i+2,j]
            if Cat[i,j+2]==Cat[i+2,j+2]:
                Cat[i+1,j+2]=Cat[i+2,j+2]
            
            if Cat[i,j]==Cat[i,j+2]==Cat[i+2,j]==Cat[i+2,j+2]:
                Cat[i+1,j+1]=Cat[i,j]
            
            # p2
            num_1=0
            num_2=0
            if Cat[i,j]==1:
                num_1+=1
            else:
                num_2+=1
            if Cat[i,j+2]==1:
                num_1+=1
            else:
                num_2+=1
            if Cat[i+2,j]==1:
                num_1+=1
            else:
                num_2+=1
            if Cat[i+2,j+2]==1:
                num_1+=1
            else:
                num_2+=1
            
            if num_1==3:
                Cat[i+1,j+1]=1
            if num_2==3:
                Cat[i+1,j+1]=2
            # ========================画边框
            if (i + 2 > a.shape[0] - 2 or j + 2 > a.shape[1] - 2):
                continue
            # 计算实际的i，j的x，y
            if flag_y[i, j] == -1:
                rho = np.abs(a[i, j]) / (np.abs(a[i, j]) + np.abs(a[i, j + 2]))
                # rho = min(max_r, max(min_r, rho))
                if np.isnan(rho).any():
                    print("flag_y[i, j]", np.abs(a[i, j]), np.abs(a[i, j]), np.abs(a[i + 2, j]))

                X[i, j + 1] = (1 - rho) * X[i, j] + rho * X[i, j + 2]
                Cat[i, j + 1] = 0  # mark as boundary

            if flag_y[i + 2, j] == -1:
                rho = np.abs(a[i + 2, j]) / (np.abs(a[i + 2, j]) + np.abs(a[i + 2, j + 2]))
                # rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print("flag_y[i, j + 2]", abs(a[i, j + 2]), abs(a[i, j + 2]), abs(a[i + 2, j + 2]))

                X[i + 2, j + 1] = (1 - rho) * X[i + 2, j] + rho * X[i + 2, j + 2]
                Cat[i + 2, j + 1] = 0  # mark as boundary

            if flag_x[i, j] == -1:
                rho = np.abs(a[i, j]) / (np.abs(a[i, j]) + np.abs(a[i + 2, j]))
                # rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print(" flag_x[i, j]", abs(a[i, j]), abs(a[i, j]), abs(a[i, j + 2]))

                Y[i + 1, j] = (1 - rho) * Y[i, j] + rho * Y[i + 2, j]
                Cat[i + 1, j] = 0  # mark as boundary

            if flag_x[i, j + 2] == -1:
                rho = np.abs(a[i, j + 2]) / (np.abs(a[i, j + 2]) + np.abs(a[i + 2, j + 2]))
                # rho = min(max_r, max(min_r, rho))

                if np.isnan(rho).any():
                    print("flag_x[i + 2, j]", abs(a[i + 2, j]), abs(a[i + 2, j]), abs(a[i + 2, j + 2]))

                Y[i + 1, j + 2] = (1 - rho) * Y[i, j + 2] + rho * Y[i + 2, j + 2]
                Cat[i + 1, j + 2] = 0  # mark as boundary

            if Cat[i, j + 1] + Cat[i + 2, j + 1] == 0:  # 上下是边界
                X[i + 1, j + 1] = (X[i, j + 1] + X[i + 2, j + 1]) * 0.5
                Cat[i + 1, j + 1] = 0

            if Cat[i + 1, j] + Cat[i + 1, j + 2] == 0:
                Y[i + 1, j + 1] = (Y[i + 1, j] + Y[i + 1, j + 2]) * 0.5
                Cat[i + 1, j + 1] = 0

    if np.isnan(X).any():
        print("you x a")
    if np.isnan(Y).any():
        print("you y a")
    return Index, X, Y, Cat


def split_tri(Tri,index_x_y_cat):
    tri_cat=[]
    for i, tri in enumerate(Tri):
        flag = 2
        for node in tri:
            # Todo 下标回去
            if (index_x_y_cat[int(node), -1] == 1):
                flag = 1
                tri_cat.append(flag)
                break
        if flag == 2:
            tri_cat.append(flag)
    return tri_cat

def load_net(path,config):
    with open(path,'rb') as f:
        genome=pickle.load(f)
        net=neat2.nn.FeedForwardNetwork.create(genome,config)
    return net,genome

def get_tri_cat(Tri,index_x_y_cat):
    cat=[]
    for i, tri in enumerate(Tri):
        flag=0
        for node in tri:
            # Todo 下标回去
            if (index_x_y_cat[int(node), -1] == 2):
                flag=2
                break
                # outside_tri.append(tri)
                # break
        if flag==0:
            cat.append(1)
        else:
            cat.append(2)
    return cat

def getshape(path,config,thresh,pcd,Tri,shapex,shapey):
    net,genome=load_net(f'{path}.pkl',config)
    pcd2=pcd.copy()
    outputs = []
    for point in pcd2:
        output = net.activate(point)
        outputs.append(output)
    outputs = np.array(outputs)
    outputs=utils.scale(outputs)
    outputs=outputs.reshape(shapey,shapex)
    Index, X, Y, Cat=find_contour(outputs,thresh,pcd2)

    x_values = X.flatten()
    y_values = Y.flatten()
    cat_values = Cat.flatten()
    index_values = Index.flatten()
    U=genome.u

    index_x_y_cat = np.concatenate(
        (index_values.reshape(-1, 1), x_values.reshape(-1, 1), y_values.reshape(-1, 1), cat_values.reshape(-1, 1)),
        axis=1)
    triangles = Tri
    x_coords = index_x_y_cat[:, 1]
    y_coords = index_x_y_cat[:, 2]
    cat=get_tri_cat(Tri,index_x_y_cat)
    plt.figure(figsize=(39, 13))
    triang = Triangulation(x_values, y_values)
    plt.triplot(x_coords, y_coords, triangles, '-', lw=0.1)
    plt.tripcolor(x_coords, y_coords, triangles,facecolors=cat, edgecolors='k',cmap='PuBu_r')
    plt.savefig(f'{path}.png')
    plt.close()
    
    # 画位移图
    displacements_x = U[::2]  # 提取位移场中的 x 方向位移
    displacements_y = U[1::2] # 提取位移场中的 y 方向位移
    x_values = x_values + displacements_x
    y_values = y_values + displacements_y

    plt.figure(figsize=(39, 13))
    triang = Triangulation(x_values, y_values)
    plt.triplot(x_values, y_values, triangles, '-', lw=0.1)
    plt.tripcolor(x_values, y_values, triangles,facecolors=cat, edgecolors='k',cmap='PuBu_r')
    plt.savefig(f'{path}_displacement.png')
    plt.close()

    return genome
    