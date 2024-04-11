##  对矩阵做SVD，并返回虚拟指标维度最小的分解方式
import scipy
from Tool.Singular2Matrix import singular2matrix


##  一般的奇异值分解，总选择较小的虚拟指标，结果是精确的
def svd(M):
    ##  调用scipy做奇异值分解
    [U, gamma, V] = scipy.linalg.svd(M)  # 对矩阵做奇异值分解
    gamma=singular2matrix(gamma,U.shape[0],V.shape[1])  # 将奇异值列表转化为奇异值矩阵

    ##  判断合并，使虚拟指标维度最小
    if U.shape[1]>V.shape[0]:
        U=U@gamma
    else:
        V=gamma@V
    return U,V


##  用于中心正交化的奇异值分解，选择某个方向合并，结果是精确的
def svd_lr(M,type,chi):
    ##  初始化U和V
    U=0
    V=0

    ##  判断左保留还是右保留
    try:
        if type=='l' or type=='L':
            [U, gamma, V] = scipy.linalg.svd(M)  # 对矩阵做奇异值分解
            V = singular2matrix(gamma, U.shape[0], V.shape[1]) @ V  # V与gamma合并
            U=U[:,:len(gamma)]  # U保留并根据gamma维度截断
            V=V[:len(gamma),:]  # V根据gamma维度截断
        elif type=='r' or type=='R':
            [U, gamma, V] = scipy.linalg.svd(M)
            U = U@singular2matrix(gamma, U.shape[0], V.shape[1])  # U与gamma合并
            U = U[:, :len(gamma)]  # U保留并根据gamma维度截断
            V = V[:len(gamma),:]  # V根据gamma维度截断
        else:
            raise Exception('类型只能为左或右')

    ##  将输入错误的类型抛出异常
    except Exception as error:
        print(error)

    ##  返回结果
    return U,V


##  用于算符作用的奇异值分解，保留一定的维度，结果是近似的
def svd_chi(M,chi):
    ##  调用scipy计算奇异值分解
    [U, gamma, V] = scipy.linalg.svd(M)  # 对矩阵做奇异值分解
    gamma=singular2matrix(gamma,U.shape[0],V.shape[1])  # 将奇异值列表转化为奇异值矩阵

    ##  保留较小的虚拟指标维度
    if U.shape[1]>V.shape[0]:
        U=U@gamma
    else:
        V=gamma@V

    ##  对虚拟指标维度截断然后返回结果
    return U[:,:chi],V[:chi,:]