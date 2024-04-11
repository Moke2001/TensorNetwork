from Basis.Operator.GeneralOperator.PauliOperator import pauli_operator
from Basis.Operator.LindbladList import LindbladList


def local_decay(N,gamma_list):
    ##  类型检查模块
    assert isinstance(gamma_list, list),"gamma_list必须是list类型"
    assert isinstance(N,int),"N必须是int类型"
    assert all(isinstance(term, int) or isinstance(term,float) for term in gamma_list),"gamma_list中的元素必须是int类型或float类型"
    assert len(gamma_list) == N,"N必须与gamma_list长度相等"

    single_list=[]  # 列表初始化

    ##  按照gamma_list中参数数据给出每个衰减形式的Lindblad算符
    for i in range(N):
        single_list.append(gamma_list[i]*pauli_operator(N,i,'-'))

    ##  返回结果
    return LindbladList('decay',N,single_list)