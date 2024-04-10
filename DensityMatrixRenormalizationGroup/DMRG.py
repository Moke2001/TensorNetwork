##  对一个量子系统求哈密顿量的基态
import numpy as np
from Basis.Operator.OperatorList import OperatorList
from Basis.State.MatrixProductState import MatrixProductState
from DensityMatrixRenormalizationGroup.EffectiveHamiltonian import effective_hamiltonian


def dmrg(H_list,chi):

    assert isinstance(H_list,OperatorList),"H_list必须是OperatorList类型"
    assert isinstance(chi,int),"chi必须是int类型"

    N=H_list.N  # 系统的局域个数
    dim_list=H_list.get_dim()  # 系统各局域的维度
    sweep_times=100  # 扫描次数
    psi=MatrixProductState.random_mps(dim_list,0,chi)  # 初始的随机MPS
    energy_result=8000  # 能量初始化
    flag=0  # 方向指标
    position=0  # 位点指标

    for i in range(sweep_times):

        ##  求有效哈密顿量，并求基态
        H_effective=effective_hamiltonian(H_list,position,psi,chi)
        dim_matrix=np.prod(psi[position].shape)
        H_effective_matrix=H_effective.reshape((dim_matrix,dim_matrix))
        eig_value, eig_vector = np.linalg.eig(H_effective_matrix)
        index_min=np.argmax(eig_value)
        moment=eig_value[index_min].reshape(psi[position].shape)
        psi[position]=moment/np.linalg.norm(moment)


        ##  变换中心位置
        if i==N-1:
            position=position-1
            flag=1
        elif i==0:
            position=position+1
            flag=0
        elif flag==0:
            position=position+1
        elif flag==1:
            position=position-1
        else:
            raise Exception("循环错误")
        psi=psi.center_orthogonalization(position)

        ##  判断是否收敛
        if np.abs(energy_result - eig_value[index_min]) < 0.001:
            return psi, eig_value[index_min]
        else:
            energy_result=eig_value[index_min]




