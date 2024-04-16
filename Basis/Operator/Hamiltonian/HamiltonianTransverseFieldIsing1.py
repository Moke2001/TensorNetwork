##  构造一维横场Ising模型的哈密顿量
from Basis.Operator.GeneralOperator.PauliOperator import pauli_operator
from Basis.Operator.OperatorList import OperatorList


def hamiltonian_transverse_field_ising1(N,J_list,h_list):
    ##  类型检查模块
    assert isinstance(N, int), "N必须是int类型"
    assert isinstance(J_list, list), "J_list必须是list类型"
    assert isinstance(h_list, list), "h_list必须是list类型"
    assert N == len(J_list) + 1 and N == len(h_list), "N必须与参数列表相对应"
    assert all(isinstance(term, float) or isinstance(term, int) for term in h_list), "h_list中的元素必须是int类型或folat类型"
    assert all(isinstance(term, float) or isinstance(term, int) for term in J_list), "J_list中的元素必须是int类型或folat类型"

    ##  算符列表初始化
    H_list_single = []
    H_list_double = []

    ##  将算符代入到每一个点位上
    for i in range(N - 1):
        if i != N - 2:
            sigmaz_0 = pauli_operator(N, i, 'z')
            sigmaz_1 = pauli_operator(N, i + 1, 'z')
            sigmax = pauli_operator(N, i, 'x')
            H_list_single.append(h_list[i] * sigmax)
            H_list_double.append(J_list[i] * sigmaz_0 * sigmaz_1)
        else:
            sigmaz_0 = pauli_operator(N, i, 'z')
            sigmaz_1 = pauli_operator(N, i + 1, 'z')
            sigmax_0 = pauli_operator(N, i, 'x')
            sigmax_1 = pauli_operator(N, i+1, 'x')
            H_list_single.append(h_list[i] * sigmax_0)
            H_list_single.append(h_list[i+1] * sigmax_1)
            H_list_double.append(J_list[i] * sigmaz_0 * sigmaz_1)

    ##  返回结果
    result = OperatorList('TransverseFieldIsing1', N, H_list_single, H_list_double)
    return result
