from Basis.Operator.GeneralOperator.PauliOperator import pauli_operator
from Basis.Operator.OperatorList import OperatorList


def hamiltonian_pxp(N, J_list, h_list, V):
    ##  类型检查模块
    assert isinstance(N, int), "N必须是int类型"
    assert isinstance(J_list, list), "J_list必须是list类型"
    assert isinstance(h_list, list), "h_list必须是list类型"
    assert N == len(J_list) and N == len(h_list), "N必须与参数列表相对应"
    assert all(isinstance(term, float) or isinstance(term, int) for term in h_list), "h_list中的元素必须是int类型或folat类型"
    assert all(isinstance(term, float) or isinstance(term, int) for term in J_list), "J_list中的元素必须是int类型或folat类型"

    ##  哈密顿量算符列表初始化
    H_single_list = []
    H_double_list = []

    ##  循环赋予算符列表算符
    for i in range(N - 1):

        ##  非最后一个位点正常处理
        if i != N - 2:
            term_0 = J_list[i] * pauli_operator(N, i, 'x')
            term_1 = h_list[i] * pauli_operator(N, i, 'z')
            term_2 = V * pauli_operator(N, i, '1') * pauli_operator(N, i + 1, '1')
            H_single_list.append(term_0 + term_1)
            H_double_list.append(term_2)

        ##  最后一个位点特殊处理
        else:
            term_0 = J_list[i] * pauli_operator(N, i, 'x')
            term_1 = h_list[i] * pauli_operator(N, i, 'z')
            term_2 = V * pauli_operator(N, i, '1') * pauli_operator(N, i + 1, '1')
            term_3 = J_list[i + 1] * pauli_operator(N, i + 1, 'x')
            term_4 = h_list[i + 1] * pauli_operator(N, i + 1, 'z')
            H_single_list.append(term_0 + term_1)
            H_single_list.append(term_3 + term_4)
            H_double_list.append(term_2)

    ##  返回结果
    result = OperatorList('PXP', N, H_single_list, H_double_list)
    return result