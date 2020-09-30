import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter


def den_proc(c):
    """
    用K近邻的方法为每个城市的density打标签
    :param c:
    :return:
    """
    density = pd.read_csv("./train_data_all/city_"+c+"/density.csv", header=None)
    print("城市的density尺寸是：", np.shape(density))
    density = np.array(density)
    grid_attr = pd.read_csv("./train_data_all/city_"+c+"/grid_attr.csv", header=None)
    grid_attr = np.array(grid_attr)
    x_train = grid_attr[:, 0:1]
    y_train = grid_attr[:, 2]
    KNN = KNeighborsClassifier(n_neighbors=5)
    KNN.fit(x_train, y_train)
    point = int(np.shape(density)[0])
    density_pre = np.zeros((point, 1))
    for i in range(15):
        a = int(point/15*i)
        b = int(point/15*(i+1))
        pred = KNN.predict(density[a:b, 2:3])
        for j in range(int(point/15)):
            density_pre[a+j, 0] = pred[j]
    print("Step1: succeed!")
    return density, density_pre


def den_cal(density, pre_loc, place_num):
    """
    den_pre: 是四维的，[日期，地区编号，人口总密度，采样的区域点数]
    :param density:
    :param pre_loc:
    :param place_num
    :return:
    """
    data_num = len(Counter(density[:, 0]))
    den_pre = np.zeros((place_num * data_num, 4))
    data_la = np.zeros((data_num, 1))
    move_num = 0
    move_s = 0
    for i in range(np.shape(density)[0]):
        if i == 0:
            index = int(float(pre_loc[i, 0]))
            den_pre[index, 2] = den_pre[index, 2] + density[i, 4]
            den_pre[index, 3] = den_pre[index, 3] + 1
            data_la[move_s, 0] = density[i, 0]
        else:
            if density[i, 0] == density[i - 1, 0]:
                index = int(float(pre_loc[i, 0]))
                den_pre[index + move_num, 2] = den_pre[index + move_num, 2] + density[i, 4]
                den_pre[index + move_num, 3] = den_pre[index + move_num, 3] + 1
            else:
                move_s += 1
                move_num = move_s * place_num
                index = int(float(pre_loc[i, 0]))
                den_pre[index + move_num, 2] = den_pre[index + move_num, 2] + density[i, 4]
                den_pre[index + move_num, 3] = den_pre[index + move_num, 3] + 1
                data_la[move_s, 0] = density[i, 0]

    flag = 0
    for i in range(np.shape(data_la)[0]):
        for j in range(place_num):
            den_pre[flag, 0] = data_la[i, 0]
            den_pre[flag, 1] = j
            flag += 1

    print("Step2: succeed!")
    return den_pre


def infection(c, loc):
    """
    为数据加入感染病人数这一特征
    :param c:
    :param loc:
    :return:
    """
    inf = pd.read_csv("./train_data_all/city_"+c+"/infection.csv", header=None)
    inf = np.array(inf.values.tolist())
    loc_inf = np.zeros((np.shape(inf)[0], 2))
    for i in range(np.shape(inf)[0]):
        for j in range(np.shape(loc)[0]):
            if int(float(loc[j, 0])) == int(inf[i, 2]):
                if int(float(loc[j, 1])) == int(inf[i, 1]):
                    loc_inf[i, 0] = loc[j, 2]
                    loc_inf[i, 1] = loc[j, 3]

    den_inf = np.hstack((inf, loc_inf))
    print("Step3: succeed!")

    return den_inf, inf


def chazhi(flag, data):
    """
    差值进去，使所有日期均有值
    :param flag:
    :param data:
    :return:
    """
    data_num = 60
    data_1 = data[:, -2]
    data_2 = data[:, -1]
    for i in range(flag):

        tmp_1 = data_1[i * data_num:(i + 1) * data_num]
        tmp_2 = data_2[i * data_num:(i + 1) * data_num]
        for j in range(data_num):
            sum_1 = 0
            sum_2 = 0
            num_1 = 0
            num_2 = 0
            if float(tmp_1[j]) == 0:  # 如果为空即插值。
                if j < 5:
                    y = tmp_1[0:j + 5 + 1]
                elif j >= 5 and j + 3 < data_num:
                    y = tmp_1[j - 2:j + 2 + 1]
                else:
                    y = tmp_1[j - 3:-1]
                for k in range(len(y)):
                    sum_1 += float(y[k])
                    if float(y[k]) == 0:
                        num_1 += 1
                if num_1 == len(y):
                    tmp_1[j] = data_1[j + (i - 1) * data_num]
                else:
                    tmp_1[j] = sum_1 / (len(y) - num_1)
            if float(tmp_2[j]) == 0:  # 如果为空即插值。
                if j < 5:
                    z = tmp_2[0:j + 5 + 1]
                elif j >= 5 and j + 3 < data_num:
                    z = tmp_2[j - 2:j + 2 + 1]
                else:
                    z = tmp_2[j - 3:-1]
                for k in range(len(z)):
                    sum_2 += float(z[k])
                    if float(z[k]) == 0:
                        num_2 += 1
                if num_2 == len(z):
                    tmp_2[j] = data_2[j + (i - 1) * data_num]
                else:
                    tmp_2[j] = sum_2 / (len(z) - num_2)

        for j in range(data_num):
            data_1[j + i * data_num] = tmp_1[j]
            data_2[j + i * data_num] = tmp_2[j]
    data_1 = data_1.reshape((np.shape(data)[0], -1))
    data_2 = data_2.reshape((np.shape(data)[0], -1))

    data_pre = np.hstack((data[:, :4], data_1, data_2))
    print("Step4: succeed!")
    return data_pre


if __name__ == "__main__":
    city = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    place = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]

    infection_all = np.zeros((2, 4))
    loc_inf_all = np.zeros((2, 6))
    for m in range(len(city)):
        den, den_label = den_proc(city[m])
        den_1 = den_cal(den, den_label, place[m])
        loc_inf_1, inf_1 = infection(city[m], den_1)
        loc_inf_2 = chazhi(place[m], loc_inf_1)
        print("完成了城市", city[m], "的数据处理")
        loc_inf_all = np.vstack((loc_inf_all, loc_inf_2))
        infection_all = np.vstack((infection_all, inf_1))

    loc_inf_all = np.delete(loc_inf_all, [0, 1], axis=0)
    infection_all= np.delete(infection_all, [0, 1], axis=0)
    np.savetxt("./loc_inf_all.csv", loc_inf_all, delimiter=",", fmt='%s')
    np.savetxt("./infection_all.csv", infection_all, delimiter=",", fmt='%s')
    print("完成了所有城市的数据处理")
