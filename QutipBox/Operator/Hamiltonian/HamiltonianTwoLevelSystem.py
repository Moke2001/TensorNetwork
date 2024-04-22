##  定义含有自发辐射的二能级系统的Rabi振荡
import os
import numpy as np
from matlab import engine
from qutip import *
from QutipBox.Operator.GeneralOperator.OperatorSigma import operator_sigma


##  耦合到算符的时变参数
def coff(t,args):
    omega_L=args['omega_L']
    omega_E=args['omega_E']
    phi=args['phi']
    return omega_E*np.cos(omega_L*t+phi)


##  哈密顿量构造
def hamiltonian_two_level_system(omega_delta,omega_L,omega_E,gamma,phi):
    ##  参数类型检查
    assert isinstance(omega_delta,float) or isinstance(omega_delta,int),"omega_delta必须是float类型或int类型"
    assert isinstance(omega_L,float) or isinstance(omega_L,int),"omega_L必须是float类型或int类型"
    assert isinstance(omega_E,float) or isinstance(omega_E,int),"omega_E必须是float类型或int类型"
    assert isinstance(gamma,float) or isinstance(gamma,int),"gamma必须是float类型或int类型"

    ##  模型构造
    H_0=operator_sigma(1,0,'z')*(omega_delta/2)  # 构造不含时的二能级原子系统的哈密顿量
    H_1=operator_sigma(1,0,'x')  # 耦合到时间项的哈密顿量部分
    args = {'omega_L': omega_L, 'omega_E': omega_E, 'phi': phi}  # 构造参数字典
    H= [H_0,[H_1,coff]]  # 构造含时哈密顿量
    C=operator_sigma(1,0,'-')*gamma  # 构造坍缩算符

    ##  返回结果
    return H,C,args


if __name__=='__main__':
    ##  构造模型
    H,C,args=hamiltonian_two_level_system(10,10,1,5,0)  # 哈密顿量，坍缩算符与参数列表
    psi=basis(2,0)  # 初态态矢
    t_list=np.linspace(0,1,1000)  # 时间区间
    E=[operator_sigma(1,0,'+')*operator_sigma(1,0,'-'),operator_sigma(1,0,'i')-operator_sigma(1,0,'+')*operator_sigma(1,0,'-')]  # 可观测量
    result=mesolve(H,psi,t_list,C,E,args).expect  # 结果

    ##  绘制图形
    eng=engine.start_matlab()
    eng.plot(t_list,result[0],'LineWidth',1.5)
    eng.hold('on',nargout=0)
    eng.plot(t_list,result[1],'LineWidth',1.5)
    eng.hold('on',nargout=0)
    eng.xlabel('Time',nargout=0)
    eng.ylabel('Number of Possessions',nargout=0)
    eng.legend(nargout=0)
    eng.grid('on',nargout=0)
    os.system('pause')