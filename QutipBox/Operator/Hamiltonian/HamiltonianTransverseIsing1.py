##  本文件构造一维横场Ising模型的Hamiltonian
from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma


##  一维Ising模型
def hamiltonian_transverse_ising1(N, J, h):

    ## 初始化系统的哈密顿量
    H = 0

    ##  循环构造每一个自旋上的算符
    for i in range(N):

        ##  当不在最后一个位点时，同时存在两种算符
        if i!=N-1:
            H_local=h[i]*operator_sigma(N,i,'x')
            H_interaction=J[i]*operator_sigma(N,i,'z')*operator_sigma(N,i+1,'z')

        ##  否则只存在一种算符
        else:
            H_local = h[i] * operator_sigma(N, i, 'x')
            H_interaction=0

        ##  将相互作用项和单体项相加
        H=H+H_local+H_interaction

    ##  返回哈密顿量给主程序
    return H


if __name__ == '__main__':
    print(hamiltonian_transverse_ising1(3, [0.5, 0.1], [0.2, 0.2, 0.3]))