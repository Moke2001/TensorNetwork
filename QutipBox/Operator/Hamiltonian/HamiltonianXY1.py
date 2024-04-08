##  本文件构造一维XY模型的Hamiltonian
from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma


def hamiltonian_xy1(N, J, h):
    ## 初始化系统的哈密顿量
    H = 0

    ##  循环构造每一个自旋上的算符
    for i in range(N):

        ##  当不在最后一个位点时，同时存在两种算符
        if i != N - 1:
            H_local = h[i] * operator_sigma(N, i, 'z')
            H_interaction_x = J[i] * operator_sigma(N, i, 'x') * operator_sigma(N, i + 1, 'x')
            H_interaction_y = J[i] * operator_sigma(N, i, 'y') * operator_sigma(N, i + 1, 'y')

        ##  否则只存在一种算符
        else:
            H_local = h[i] * operator_sigma(N, i, 'z')
            H_interaction_x = 0
            H_interaction_y = 0

        ##  将相互作用项和单体项相加
        H = H + H_local + H_interaction_x+ H_interaction_y

    ##  返回哈密顿量给主程序
    return H
