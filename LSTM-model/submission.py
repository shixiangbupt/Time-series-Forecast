import pandas as pd
import numpy as np


result_A = pd.read_csv("./data/city_A_result.csv", header=None)
result_A = np.array(result_A.values.tolist())
la = np.shape(result_A)[1]

submission = np.zeros((392 * 30, 3))
city = []
for i in range(np.shape(result_A)[1]):
    data = 21200615
    for j in range(30):
        city.append('A')
        submission[j + 30 * i, 0] = int(i)
        if data > 21200630:
            submission[j + 30 * i, 1] = int(data + 70)
        else:
            submission[j + 30 * i, 1] = int(data)
        data += 1
        submission[j + 30 * i, 2] = int(result_A[j, i])


result_B = pd.read_csv("./data/city_B_result.csv", header=None)
result_B = np.array(result_B.values.tolist())
lb = np.shape(result_B)[1]


for i in range(np.shape(result_B)[1]):
    data = 21200615
    for j in range(30):
        city.append('B')
        submission[j + 30 * (i + la), 0] = int(i)
        if data > 21200630:
            submission[j + 30 * (i + la), 1] = int(data + 70)
        else:
            submission[j + 30 * (i + la), 1] = int(data)
        data += 1
        submission[j + 30 * (i + la), 2] = int(result_B[j, i])


result_C = pd.read_csv("./data/city_C_result.csv", header=None)
result_C = np.array(result_C.values.tolist())
lc = np.shape(result_C)[1]


for i in range(np.shape(result_C)[1]):
    data = 21200615
    for j in range(30):
        city.append('C')
        submission[j + 30 * (i + la + lb), 0] = int(i)
        if data > 21200630:
            submission[j + 30 * (i + la + lb), 1] = int(data + 70)
        else:
            submission[j + 30 * (i + la + lb), 1] = int(data)
        data += 1
        submission[j + 30 * (i + la + lb), 2] = int(result_C[j, i])


result_D = pd.read_csv("./data/city_D_result.csv", header=None)
result_D = np.array(result_D.values.tolist())
ld = np.shape(result_D)[1]


for i in range(np.shape(result_D)[1]):
    data = 21200615
    for j in range(30):
        city.append('D')
        submission[j + 30 * (i + la + lb + lc), 0] = int(i)
        if data > 21200630:
            submission[j + 30 * (i + la + lb + lc), 1] = int(data + 70)
        else:
            submission[j + 30 * (i + la + lb + lc), 1] = int(data)
        data += 1
        submission[j + 30 * (i + la + lb + lc), 2] = int(result_D[j, i])


result_E = pd.read_csv("./data/city_E_result.csv", header=None)
result_E = np.array(result_E.values.tolist())
le = np.shape(result_D)[1]


for i in range(np.shape(result_E)[1]):
    data = 21200615
    for j in range(30):
        city.append('E')
        submission[j + 30 * (i + la + lb + lc + ld), 0] = int(i)
        if data > 21200630:
            submission[j + 30 * (i + la + lb + lc + ld), 1] = int(data + 70)
        else:
            submission[j + 30 * (i + la + lb + lc + ld), 1] = int(data)
        data += 1
        submission[j + 30 * (i + la + lb + lc + ld), 2] = int(result_E[j, i])

city = np.array(city)
city = np.reshape(city, (11760, 1))
print(np.shape(submission))
print(np.shape(city))
submission = np.hstack((city, submission))
print(submission)

result = pd.DataFrame({'city': submission[:, 0], 'loc': submission[:, 1], 'data': submission[:, 2],
                       'num': submission[:, 3]})

# 将DataFrame存储为csv,index表示是否显示行名，default=True
result.to_csv("./data/submission.csv", index=False, sep=',')
