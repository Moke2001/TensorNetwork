from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma


def hamiltonian_creator(N,J_list,h_list,V):
    H=0
    for i in range(N - 1):
        if i != N - 2:
            term_0 = J_list[i] * operator_sigma(N, i, 'x')
            term_1 = h_list[i] * operator_sigma(N, i, 'z')
            term_2 = V * operator_sigma(N, i, '1') * operator_sigma(N, i + 1, '1')
            H=H+term_0 + term_1 + term_2
        else:
            term_0 = J_list[i] * operator_sigma(N, i, 'x')
            term_1 = h_list[i] * operator_sigma(N, i, 'z')
            term_2 = V * operator_sigma(N, i, '1') * operator_sigma(N, i + 1, '1')
            term_3 = J_list[i + 1] * operator_sigma(N, i + 1, 'x')
            term_4 = h_list[i + 1] * operator_sigma(N, i + 1, 'z')
            H=H+term_0 + term_1 + term_2+term_3+term_4

    ##  返回结果
    return H
