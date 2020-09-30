import numpy as np
import pandas as pd


city = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
place = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]

for m in range(len(city)):
    mig = pd.read_csv("./train_data_all/city_"+city[m]+"/migration.csv", header=None)
    mig = np.array(mig.values.tolist())
    mig_pre = np.zeros((60, 3))
    str1 = city[m]
    flag = 0
    for i in range(np.shape(mig)[0]):
        if i == 0:
            if mig[i, 1] == str1:
                mig_pre[flag, 0] = float(mig[i, 3])
                mig_pre[flag, 2] = mig[i, 0]
            else:
                mig_pre[flag, 1] = float(mig[i, 3])
                mig_pre[flag, 2] = mig[i, 0]
        elif mig[i, 0] == mig[i - 1, 0]:
            if mig[i, 1] == str1:
                mig_pre[flag, 0] = float(mig[i, 3]) + mig_pre[flag, 0]
            else:
                mig_pre[flag, 1] = float(mig[i, 3]) + mig_pre[flag, 1]
        else:
            flag += 1
            if mig[i, 1] == str1:
                mig_pre[flag, 0] = float(mig[i, 3])
                mig_pre[flag, 2] = mig[i, 0]
            else:
                mig_pre[flag, 1] = float(mig[i, 3])
                mig_pre[flag, 2] = mig[i, 0]

    inf = pd.read_csv("./train_data_all/city_"+str1+"/infection.csv", header=None)
    inf = np.array(inf.values.tolist())
    loc_num = place[m]
    inf_sum = np.zeros((60, 1))
    for j in range(np.shape(inf)[0]):
        for i in range(45):
            if inf[i, 2] == inf[j, 2]:
                inf_sum[i] = inf_sum[i] + float(inf[j, 3])

    city_inf = pd.DataFrame({'data': mig_pre[:, 2], 'out': mig_pre[:, 0], 'in': mig_pre[:, 1],
                             'infection': inf_sum[:, 0]})
    city_inf.to_csv("city"+str1+"_wcx.csv", index=False, sep=',')
    print('城市', str1, '的数据已经处理完毕！')


