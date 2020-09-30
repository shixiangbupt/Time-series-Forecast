import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, LSTM
import os


def data_pre(loc_num):
    """
    该函数用于特征提取，包括城市的迁入量迁出量、地区的迁入迁出、地区内部的人口流动、地区每日新增的感染人数
    -loc_num:该城市的地区数目
    :return: x_train, y_train, x_test, y_test
    """
    # 导入城市间的迁移量特征(前五维是城市的新增患者数，6-10维是五个城市的迁出量，11-15维是五个城市的迁入量)
    data = pd.read_csv("./data/city_all_norm.csv", header=None)
    data = np.array(data.values.tolist())
    # index = np.array(list(range(0, 45)))  # 用于可视化的横轴
    print(np.shape(data))

    # 导入地区间的迁移特征
    migration = pd.read_csv("./data/A_migration.csv", header=None)
    migration = np.array(migration.values.tolist())
    print(np.shape(migration))
    # 将迁移数据处理成每个地区的三维特征[内部流动、迁入、迁出]，维度为[日期，3*地区数]
    pre_mig = np.zeros((45, 3*loc_num))
    flag = 0  # 代替日期
    for j in range(np.shape(migration)[0]):
        if j != 0:
            if migration[j-1, 0] != migration[j, 0]:
                flag += 1
        for i in range(loc_num):
            if int(migration[j, 1]) == i:
                if int(migration[j, 2]) == i:
                    pre_mig[flag, i*3] = migration[j, 3]
                else:
                    pre_mig[flag, 2+i*3] = pre_mig[flag, 2+i*3] + migration[j, 3]
            if int(migration[j, 2]) == i and int(migration[j, 1]) != i:
                pre_mig[flag, 1+i*3] = pre_mig[flag, 1+i*3] + migration[j, 3]
    print(pre_mig)
    print(np.shape(pre_mig))

    # 导入各地区的人口密度
    loc_pre = np.zeros((45*loc_num, 1))
    loc = pd.read_csv("./data/featureA_ljq.csv")
    loc = np.array(loc.values.tolist())
    for i in range(45*loc_num):
        loc_pre[i] = round(float(loc[i, 2])/float(loc[i, 5]), 4)
    print(loc)

    # 构建模型所需要的训练集和测试集
    dataset = np.zeros((45*loc_num, 7))
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    # 构造标准数据集(维度[45*loc_num,7])
    for j in range(loc_num):
        for i in range(45):
            dataset[i+j*45, 0] = data[i, 5]  # 需要按照城市修改
            dataset[i+j*45, 1] = data[i, 10]  # 需要按照城市修改
            dataset[i+j*45, 2] = norm(pre_mig[i, j*3])
            dataset[i+j*45, 3] = norm(pre_mig[i, 1+j*3])
            dataset[i+j*45, 4] = norm(pre_mig[i, 2+j*3])
            dataset[i+j*45, 5] = norm(loc_pre[i+j*45])
            dataset[i+j*45, 6] = norm(float(loc[i+j*45, 3]))
    # 划分训练集和测试集(用前五天的数据预测后一天的数据)
    for i in range(loc_num):
        for k in range(5, 35):
            x_train.append(dataset[k-5+i*45:k+i*45, :])
            y_train.append(dataset[k+i*45, :])
        for k in range(36, 45):
            x_test.append(dataset[k-5+i*45:k+i*45, :])
            y_test.append(dataset[k+i*45, :])

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2]))
    print('x_train:', x_train.shape)
    x_test, y_test = np.array(x_test), np.array(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2]))
    print('x_test:', x_test.shape)

    return x_train, y_train, x_test, y_test, dataset


def norm(x):
    """
    对部分特征进行log归一化处理
    :param x: 需要归一化的输入特征
    :return: 归一化后的数据
    """
    return (math.log(x+1) + 2) / 10


def denorm(x):
    """
    对输出的日感染病人数进行恢复
    :param x: 需要去归一化的输出特征
    :return: 日感染病人数(整数)
    """
    return round(math.exp(x * 10 - 2))


def lstm_model():
    """
    构建整个LSTM预测网络
    :return: 整个预测网络
    """
    model = tf.keras.Sequential([
        LSTM(64, activation='relu', return_sequences=False, unroll=True, input_shape=(5, 7)),
        Dense(32, activation='relu'),
        Dense(7, activation='sigmoid')])
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    return model


if __name__ == '__main__':

    loc_num = 118
    xtrain, ytrain, xtest, ytest,data_all = data_pre(loc_num)
    model = lstm_model()

    # 训练整个网络模型
    checkpoint_save_path = "./checkpoint/nCov.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('-------------load the model-----------------')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     # monitor='loss',
                                                     # save_best_only=True,
                                                     verbose=1)
    history = model.fit(xtrain, ytrain, epochs=1000, batch_size=512, callbacks=[cp_callback],
                        validation_data=(xtest, ytest), validation_split=1, verbose=1)

    model.summary()

    # 可视化损失
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(1)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()

    predicted_train_data = model.predict(xtrain)
    print(predicted_train_data)
    plt.figure(2)
    plt.plot(ytrain[:, -1], label='Real infections')
    plt.plot(predicted_train_data[:, -1], label='Predicted infections')
    plt.legend(loc='upper left')
    plt.title('Predicted and Real')
    plt.show()

    # 预测后三十天的日感染病人数
    pre_label = np.zeros((30, loc_num))
    for i in range(loc_num):
        pre_set = data_all[40+i*45:45*(i+1), :]
        delay = data_all[40+i*45:45*(i+1), :]
        for j in range(30):
            pre_set = np.reshape(delay, (1, 5, 7))
            prediction = model.predict(pre_set)
            pre_label[j, i] = denorm(prediction[0, -1])
            delay = np.delete(delay, 0, axis=0)
            delay = np.vstack((delay, prediction))

    np.savetxt("./data/city_A_result.csv", pre_label, delimiter=",")
