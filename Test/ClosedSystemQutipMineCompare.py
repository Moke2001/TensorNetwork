##  将TEBD算法和精确对角化算法求系统演化的结果相比较
import numpy as np
from qutip import *
from Basis.Operator.GeneralOperator.PauliOperator import pauli_operator
from Basis.Operator.Hamiltonian.HamiltonianTransverseFieldIsing1 import hamiltonian_transverse_field_ising1
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.State.MatrixProductState import MatrixProductState
from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma
from QutipBox.Operator.Hamiltonian.HamiltonianTransverseIsing1 import hamiltonian_transverse_ising1
from TimeEvolveBlockDecimation.TEBD import tebd
from matlab import engine
import os


def compare():
    ##  基本参数设置模块-------------------------------------------------------------------------------------------------------------------

    t_0=0  # 开始时间
    t_1=20  # 结束时间
    delta_t=0.001  # 时间间隔
    chi=6  # 截断维度

    ##  实例构造模块-----------------------------------------------------------------------------------------------------------------------

    ##  TEBD算法实例构造
    state_me = np.array([[[[1, 0],[0,0]], [[0, 0],[0,0]]], [[[0, 0],[0,0]], [[0, 0],[0,0]]]])  # 初态态矢张量
    psi_me = MatrixProductState.tensor2mps(state_me, chi)  # 初态态矢MPS
    H_list_me = hamiltonian_transverse_field_ising1(4, [0.2, 0.2,0.2], [0.5, 0.5, 0.5,0.5])  # 哈密顿量
    expect_opeartor_me = H_list_me  # 可观测量

    ##  Qutip实例构造
    H_qutip=hamiltonian_transverse_ising1(4, [0.2, 0.2,0.2], [0.5, 0.5, 0.5,0.5])  # 哈密顿量
    psi_qutip=tensor(basis(2,0),basis(2,0),basis(2,0),basis(2,0))  # 初态态矢

    ##  求解模块----------------------------------------------------------------------------------------------------------------------------

    value_list_me, t_list = tebd(psi_me, H_list_me, expect_opeartor_me, t_0, t_1, delta_t, chi)  # TEBD算法求解
    value_list_qutip=mesolve(H_qutip,psi_qutip,t_list,e_ops=[H_qutip]).expect[0]  # 精确对角化求解

    ##  绘图模块----------------------------------------------------------------------------------------------------------------------------

    eng=engine.start_matlab()
    eng.plot(t_list, value_list_qutip, 'LineWidth', 1.5)
    eng.hold('on',nargout=0)
    eng.plot(t_list,value_list_me,'LineWidth',1.5)
    eng.grid('on',nargout=0)
    eng.xlabel('Time')
    eng.ylabel('Expect Value')
    os.system('pause')


if __name__ == '__main__':
    compare()

