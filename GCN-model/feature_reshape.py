import numpy as np
import pandas as pd


city = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
loc_num = [118, 30, 135, 75, 34, 331, 38, 53, 33, 8, 48]
embedding_reshape = np.zeros((903, 9))
city_n = []

index = 0
for m in range(len(city)):
    embedding = pd.read_csv('./result/city_' + city[m] + '_embedding.csv', header=None)
    embedding = np.array(embedding.values.tolist())
    for i in range(loc_num[m]):
        city_n.append(city[m])
        embedding_reshape[index, 0] = i
        for j in range(8):
            embedding_reshape[index, j + 1] = embedding[i, j]
        index += 1
city_n = np.array(city_n)
city_n = city_n.reshape(903, 1)
embedding_reshape = np.hstack((city_n, embedding_reshape))
np.savetxt('./embedding_output.csv', embedding_reshape, delimiter=",", fmt='%s')
