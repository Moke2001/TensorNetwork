##  将奇异值列表改写为矩阵的形式
import numpy as np


def singular2matrix(gamma,length_0,length_1):
    ##  矩阵初始化
    result=np.zeros((length_0,length_1))

    ##  将奇异值循环代入对角
    for i in range(len(gamma)):
        result[i][i]=gamma[i]

    ##  返回结果
    return result