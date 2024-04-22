##  计算有限温度下的平衡态密度矩阵
import numpy as np
from Basis.Operator.Hamiltonian.HamiltonianTransverseFieldIsing1 import hamiltonian_transverse_field_ising1
from Basis.Operator.OperatorList import OperatorList
from Basis.Relation.OperateOperator import operate_operator
from Basis.State.MatrixProductOperator import MatrixProductOperator


def es(H_list,T):
    ##  参数检查模块-----------------------------------------------------------------------------------------------------------------------

    assert isinstance(H_list,OperatorList),"H_list必须是OperatorList类型"
    assert isinstance(T,int) or isinstance(T,float),"T必须是int类型或float类型"

    ##  初始化模块--------------------------------------------------------------------------------------------------------------------------

    delta_tau=0.001  # 虚时间精度
    K=int(T/delta_tau)  # 循环次数
    U_list=H_list.virtual_time_evolve_operator(delta_tau)  # 虚时间演化算符
    chi = 5  # 截断维度

    ##  构造初始单位直积态
    rho = []
    dim_list=H_list.get_dim()
    for i in range(H_list.N):
        rho.append(np.identity(dim_list[i]))
    rho=MatrixProductOperator.straight2mpo(rho)

    ##  核心算法模块-----------------------------------------------------------------------------------------------------------------------

    for i in range(K):
        print(i)
        rho=operate_operator(U_list,operate_operator(U_list.dagger(),rho,chi,'r'),chi,'l')

    ##  结果返回模块-----------------------------------------------------------------------------------------------------------------------

    ##  返回结果
    return rho


if __name__=="__main__":
    H_list_test=hamiltonian_transverse_field_ising1(4,[0.5]*3,[1]*4)
    rho_test=es(H_list_test,15)
    print(rho_test)