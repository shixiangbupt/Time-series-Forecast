import numpy as np
import pandas as pd


def read_transfer(c):
    mig = pd.read_csv("./train_data_all/city_" + c + "/migration.csv", header=None)
    mig = np.array(mig.values.tolist())
    mig_pre = np.zeros((60, 3))  # [流出，流入，日期]
    str1 = c
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
    transfer = pd.read_csv("./train_data_all/tranfer_a_day_"+c+".csv")
    transfer = np.array(transfer.values.tolist())

    migration = np.zeros((60*np.shape(transfer)[0], 4))
    for i in range(60):
        for j in range(np.shape(transfer)[0]):
            migration[j+i*np.shape(transfer)[0], 0] = mig_pre[i, 2]
            migration[j + i * np.shape(transfer)[0], 1] = transfer[j, 1].replace(c, '').replace('_', '')
            migration[j + i * np.shape(transfer)[0], 2] = transfer[j, 2].replace(c, '').replace('_', '')
            migration[j + i * np.shape(transfer)[0], 3] = float(transfer[j, 3])*mig_pre[i, 1]

    np.savetxt("./"+c+"_migration.csv", migration, delimiter=",", fmt='%s')


if __name__ == "__main__":
    city = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
    for m in range(len(city)):
        read_transfer(city[m])








