##  在这里构造自定义的系统的哈密顿量
from Basis.Operator.GeneralOperator.PauliOperator import pauli_operator
from Basis.Operator.OperatorList import OperatorList


def hamiltonian_creator(N,J_list,h_list,V):
    H_single_list=[]
    H_double_list=[]
    for i in range(N-1):
        if i!=N-2:
            term_0=J_list[i]*pauli_operator(N,i,'x')
            term_1=h_list[i]*pauli_operator(N,i,'z')
            term_2=V*pauli_operator(N,i,'1')*pauli_operator(N,i+1,'1')
            H_single_list.append(term_0+term_1)
            H_double_list.append(term_2)
        else:
            term_0 = J_list[i] * pauli_operator(N, i, 'x')
            term_1 = h_list[i] * pauli_operator(N, i, 'z')
            term_2 = V * pauli_operator(N, i, '1') * pauli_operator(N, i + 1, '1')
            term_3 = J_list[i+1] * pauli_operator(N, i+1, 'x')
            term_4 = h_list[i + 1] * pauli_operator(N, i + 1, 'z')
            H_single_list.append(term_0 + term_1)
            H_single_list.append(term_3 + term_4)
            H_double_list.append(term_2)
    result=OperatorList('PXP',N,H_single_list,H_double_list)

    ##  返回结果
    return result