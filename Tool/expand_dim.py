import numpy as np


def expand_dim(U,dim_0,dim_1):

    result=np.zeros((dim_0,dim_1))
    for i in range(dim_0):
        for j in range(dim_1):
            if i<U.shape[0] and j<U.shape[1]:
                result[i][j]=U[i][j]
            else:
                result[i][j]=0

    return result