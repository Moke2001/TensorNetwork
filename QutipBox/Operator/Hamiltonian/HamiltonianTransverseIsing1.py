##  本文件构造一维横场Ising模型的Hamiltonian
import numpy as np
from matlab import engine
import os
from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma
from qutip import *

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
    H=hamiltonian_transverse_ising1(2, [0.5], [1,1])
    psi_0=tensor(basis(2,0),basis(2,0))
    psi_e=tensor(basis(2,1),basis(2,1))
    psi_1=tensor(basis(2,1),basis(2,0))
    psi_2 = tensor(basis(2, 0), basis(2, 1))
    C=[10*psi_0*psi_e.dag()]
    E=[psi_0*psi_0.dag(),psi_1*psi_1.dag(),psi_e*psi_e.dag()]
    t_list = np.linspace(0, 2, 1000)  # 时间区间
    result = mesolve(H, psi_0, t_list, C, E).expect  # 结果

    ##  绘制图形
    eng = engine.start_matlab()
    eng.plot(t_list, result[0], 'LineWidth', 1.5)
    eng.hold('on', nargout=0)
    eng.plot(t_list, result[1], 'LineWidth', 1.5)
    eng.hold('on', nargout=0)
    eng.plot(t_list, result[2], 'LineWidth', 1.5)
    eng.hold('on', nargout=0)
    eng.xlabel('Time', nargout=0)
    eng.ylabel('Number of Possessions', nargout=0)
    eng.legend(nargout=0)
    eng.grid('on', nargout=0)
    os.system('pause')