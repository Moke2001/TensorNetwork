##  输入一个MPS、位点位置和哈密顿量，求DMRG过程中的有效哈密顿量
import numpy as np
from Basis.Operator.OperatorList import OperatorList
from DensityMatrixRenormalizationGroup.MatrixTerm import matrix_term


def effective_hamiltonian(H_list, position, psi_origin):
    ##  类型检查模块-----------------------------------------------------------------------------------------------------------------------

    assert isinstance(H_list, OperatorList), "H_list必须是OperatorList类型"
    assert isinstance(position, int), "position必须是int类型"

    ##  初始化模块-------------------------------------------------------------------------------------------------------------------------

    psi_moment_down=psi_origin.copy()  # 防止改变参量
    psi_moment_up = psi_origin.copy()  # 防止改变参量
    data=psi_origin.data[position]  # 提取目标位点的数据

    ##  核心算法模块-----------------------------------------------------------------------------------------------------------------------

    ##  如果位置在头结点，构造基张量和有效哈密顿量的初始形式
    if position == 0:
        psi_down = np.zeros((data.shape[0], data.shape[1]))
        psi_up = np.zeros((data.shape[0], data.shape[1]))
        H_effective = np.zeros((data.shape[0], data.shape[1], data.shape[0], data.shape[1]), dtype=complex)

    ##  如果位置在尾结点，构造基张量和有效哈密顿量的初始形式
    elif position == H_list.N - 1:
        psi_down = np.zeros((data.shape[0], data.shape[1]))
        psi_up = np.zeros((data.shape[0], data.shape[1]))
        H_effective = np.zeros((data.shape[0], data.shape[1], data.shape[0], data.shape[1]), dtype=complex)

    ##  如果位置在中间，构造基张量和有效哈密顿量的初始形式
    else:
        psi_down = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        psi_up = np.zeros((data.shape[0], data.shape[1], data.shape[2]))
        H_effective = np.zeros((data.shape[0], data.shape[1], data.shape[2], data.shape[0], data.shape[1], data.shape[2]), dtype=complex)

    ##  当位置在首尾时，有效哈密顿量含有四个指标
    if position == 0 or position == H_list.N - 1:
        for i in range(H_effective.shape[0]):
            for j in range(H_effective.shape[1]):
                for k in range(H_effective.shape[2]):
                    for l in range(H_effective.shape[3]):
                        psi_down[i][j] = 1  # 赋予一个基张量
                        psi_up[k][l] = 1  # 赋予一个基张量
                        psi_moment_down[position]=psi_down  # 将基张量代入
                        psi_moment_up[position] = psi_up  # 将基张量代入
                        H_effective[i][j][k][l] = matrix_term(H_list, psi_moment_down, psi_moment_up)  # 求两个基张量下的矩阵元
                        psi_moment_down[position] = psi_origin[position].copy()  # 将MPS还原
                        psi_moment_up[position] = psi_origin[position].copy()  # 将MPS还原
                        psi_down[i][j] = 0  # 将基张量还原
                        psi_up[k][l] = 0  # 将基张量还原

    ##  当位置在中间时，有效哈密顿量有六个指标
    else:
        for i in range(H_effective.shape[0]):
            for j in range(H_effective.shape[1]):
                for k in range(H_effective.shape[2]):
                    for l in range(H_effective.shape[3]):
                        for p in range(H_effective.shape[4]):
                            for w in range(H_effective.shape[5]):
                                psi_down[i][j][k] = 1  # 赋予一个基张量
                                psi_up[l][p][w] = 1  # 赋予一个基张量
                                psi_moment_down[position] = psi_down  # 将基张量代入
                                psi_moment_up[position] = psi_up  # 将基张量代入
                                H_effective[i][j][k][l][p][w] = matrix_term(H_list, psi_moment_down, psi_moment_up)  # 求两个基张量下的矩阵元
                                psi_moment_down[position] = psi_origin[position].copy()  # 将MPS还原
                                psi_moment_up[position] = psi_origin[position].copy()  # 将MPS还原
                                psi_down[i][j][k] = 0  # 将基张量还原
                                psi_up[l][p][w] = 0  # 将基张量还原

    ##  结果返回模块-----------------------------------------------------------------------------------------------------------------------

    ##  返回结果
    return H_effective
