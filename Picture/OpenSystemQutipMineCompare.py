##  比较开放系统演化qutip和TEBD算法计算的区别
import numpy as np
from matlab import engine
from Basis.Operator.LindbladList import LindbladList
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.State.MatrixProductState import MatrixProductState
from OpenSystemEvolve.QuantumTrajectory import quantum_trajectory
from qutip import *
from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma
from QutipBox.Operator.Hamiltonian.HamiltonianTransverseIsing1 import hamiltonian_transverse_ising1
import os

def compare():
    ##  基本参数设置模块-------------------------------------------------------------------------------------------------------------------

    t_0=0  # 初始时刻
    t_1=15  # 结束时刻
    delta_t=0.1  # 时间间隔

    ##  实例构造模块-----------------------------------------------------------------------------------------------------------------------

    ##  TEBD算法实例构造
    state_me = np.array([[[1,0],[0,0]],[[0,0],[0,0]]])  # 初始态的张量形式
    psi_me = MatrixProductState.tensor2mps(state_me, 3)  # 初始态的MPS形式
    H_list_me=OperatorList.hamiltonian_transverse_ising1(3,[0.2,0.2],[0.5,0.5,0.5])  # 哈密顿量张量
    C_list_me=LindbladList.local_decay(3,[0.2,0.2,0.2])  # 坍缩算符张量
    expect_operator_me=Operator.operator_sigmaz(3,0)  # 可观测量算符张量

    ##  Qutip实例构造
    psi_qutip=tensor(basis(2,0),basis(2,0),basis(2,0))  # 初始态qutip对象
    H_qutip = hamiltonian_transverse_ising1(3,[0.2,0.2],[0.5,0.5,0.5])  # 哈密顿量qutip对象
    C_list_qutip=[]  # 坍缩算符qutip对象初始化
    for i in range(3):
        C_list_qutip.append(0.2*operator_sigma(3,i,'-'))
    expect_operator_qutip = operator_sigma(3,0,'z')  # 可观测量qutip对象

    ##  求解模块----------------------------------------------------------------------------------------------------------------------------

    result_me,t_list = quantum_trajectory(psi_me, H_list_me, C_list_me, expect_operator_me, t_0, t_1, delta_t)  # 量子轨迹TEBD算法求解
    result_qutip = mesolve(H_qutip,psi_qutip,t_list,C_list_qutip,[expect_operator_qutip])  # 精确对角化求解
    result_qutip = result_qutip.expect[0]  # 得到精确对角化的时间序列

    ##  绘图模块----------------------------------------------------------------------------------------------------------------------------

    eng=engine.start_matlab()
    eng.plot(t_list,result_me,'LineWidth',1.5)
    eng.hold('on',nargout=0)
    eng.plot(t_list,result_qutip,'LineWidth',1.5)
    eng.grid('on',nargout=0)
    eng.xlabel('Time')
    eng.ylabel('Expect Value')
    os.system('pause')


if __name__ == '__main__':
    compare()