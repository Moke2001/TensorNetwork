##  对一个量子系统求哈密顿量的基态
import numpy as np

from Basis.Operator.Hamiltonian.HamiltonianTransverseFieldIsing1 import hamiltonian_transverse_field_ising1
from Basis.Operator.OperatorList import OperatorList
from Basis.Relation.Expect import expect
from Basis.State.MatrixProductState import MatrixProductState
from DensityMatrixRenormalizationGroup.EffectiveHamiltonian import effective_hamiltonian
from DensityMatrixRenormalizationGroup.MatrixTerm import matrix_term


def dmrg(H_list,chi):
    ##  参量检查模块-----------------------------------------------------------------------------------------------------------------------

    assert isinstance(H_list,OperatorList),"H_list必须是OperatorList类型"
    assert isinstance(chi,int),"chi必须是int类型"

    ##  初始化模块-------------------------------------------------------------------------------------------------------------------------
    N=H_list.N  # 系统的局域个数
    dim_list=H_list.get_dim()  # 系统各局域的维度
    sweep_times=5000  # 扫描次数
    psi=MatrixProductState.random_mps(dim_list,0,chi)  # 初始的随机MPS
    energy_result=8000  # 能量初始化
    flag=0  # 方向指标
    position=0  # 位点指标

    ##  核心算法模块-----------------------------------------------------------------------------------------------------------------------

    ##  经历若干次扫描
    for i in range(sweep_times*N):

        ##  求有效哈密顿量
        H_effective=effective_hamiltonian(H_list,position,psi)  # 有效哈密顿量的张量形式
        dim_matrix=np.prod(psi[position].shape)  # 有效哈密顿量的矩阵维度
        H_effective_matrix=H_effective.reshape((dim_matrix,dim_matrix))  # 有效哈密顿量的张量形式

        ##  求有效哈密顿量的基态
        eig_value, eig_vector = np.linalg.eig(H_effective_matrix)  # 本征值和本征矢量
        index_min=np.argmin(eig_value)  # 最小本征值序号
        moment=eig_vector[:,index_min].reshape(psi[position].shape)  # 更新参量
        psi[position]=moment/np.linalg.norm(moment)  # 将结果赋予MPS

        ##  变换中心位置
        if position==N-1:
            position=position-1
            flag=1
        elif position==0:
            position=position+1
            flag=0
        elif flag==0:
            position=position+1
        elif flag==1:
            position=position-1
        else:
            raise Exception("循环错误")
        psi.center_orthogonalization(position)  # 将MPS向下一个位点中心正交化

        ##  判断是否收敛
        if np.abs(energy_result - eig_value[index_min].real) < 0.00001:
            return psi, eig_value[index_min].real
        else:
            energy_result=eig_value[index_min].real

    ##  结果处理模块-----------------------------------------------------------------------------------------------------------------------
    print('已到达循环上限')
    return psi, energy_result


if __name__=='__main__':
    H_list=hamiltonian_transverse_field_ising1(5,[0.5,0.5,0.5,0.5],[1,1,1,1,1])
    psi,energy_result=dmrg(H_list,15)
    print(energy_result)



