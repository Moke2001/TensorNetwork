##  本文件构造一维Ising模型的Hamiltonian
from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma


def hamiltonian_ising1(N, J_list, h_list):
    ##  参量检查模块-----------------------------------------------------------------------------------------------------------------------

    assert isinstance(N,int),"N必须是int类型"
    assert isinstance(J_list, list),"J_list必须是list类型"
    assert isinstance(h_list, list),"h_list必须是list类型"
    assert all(isinstance(term,float) or isinstance(term,int) for term in J_list),"J_list中的元素必须是int类型或float类型"
    assert all(isinstance(term, float) or isinstance(term, int) for term in h_list), "h_list中的元素必须是int类型或float类型"

    ##  核心计算模块-----------------------------------------------------------------------------------------------------------------------

    ## 初始化系统的哈密顿量
    H=0

    ##  循环构造每一个自旋上的算符
    for i in range(N):

        ##  当不在最后一个位点时，同时存在两种算符
        if i!=N-1:
            H_local= h_list[i] * operator_sigma(N, i, 'z')
            H_interaction= J_list[i] * operator_sigma(N, i, 'z') * operator_sigma(N, i + 1, 'z')

        ##  否则只存在一种算符
        else:
            H_local = h_list[i] * operator_sigma(N, i, 'z')
            H_interaction=0

        ##  将相互作用项和单体项相加
        H=H+H_local+H_interaction

    ##  返回哈密顿量给主程序
    return H
