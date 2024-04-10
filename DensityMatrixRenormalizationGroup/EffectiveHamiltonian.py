import numpy as np

from Basis.Operator.OperatorList import OperatorList
from DensityMatrixRenormalizationGroup.MatrixTerm import matrix_term


def effective_hamiltonian(H_list, position, psi_origin,chi):
    assert isinstance(H_list, OperatorList), "H_list必须是OperatorList类型"
    assert isinstance(position, int), "position必须是int类型"
    assert isinstance(chi, int), "chi必须是int类型"

    psi_moment_down=psi_origin.copy()
    psi_moment_up = psi_origin.copy()

    if position == 0:
        psi_down = np.zeros((H_list.single_list[0].dim, chi))
        psi_up = np.zeros((H_list.single_list[0].dim, chi))
        H_effective = np.zeros((H_list.single_list[0].dim, chi, H_list.single_list[0].dim, chi))
    elif position == H_list.N - 1:
        psi_down = np.zeros((chi, H_list.single_list[0].dim))
        psi_up = np.zeros((chi, H_list.single_list[0].dim))
        H_effective = np.zeros((chi, H_list.single_list[0].dim, chi, H_list.single_list[0].dim))
    else:
        psi_down = np.zeros((chi, H_list.single_list[0].dim, chi))
        psi_up = np.zeros((chi, H_list.single_list[0].dim, chi))
        H_effective = np.zeros((chi, H_list.single_list[0].dim, chi, chi, H_list.single_list[0].dim, chi))

    ##  构造矩阵元
    if position == 0 or position == H_list.N - 1:
        for i in range(H_effective.shape[0]):
            for j in range(H_effective.shape[1]):
                for k in range(H_effective.shape[2]):
                    for l in range(H_effective.shape[3]):
                        psi_down[i][j] = 1
                        psi_up[k][l] = 1
                        psi_moment_down[position]=psi_down
                        psi_moment_up[position] = psi_down
                        H_effective[i][j][k][l] = matrix_term(H_list, psi_moment_down, psi_moment_up)
                        psi_down[i][j] = 0
                        psi_up[k][l] = 0
    else:
        for i in range(H_effective.shape[0]):
            for j in range(H_effective.shape[1]):
                for k in range(H_effective.shape[2]):
                    for l in range(H_effective.shape[3]):
                        for p in range(H_effective.shape[4]):
                            for w in range(H_effective.shape[5]):
                                psi_down[i][j][k] = 1
                                psi_up[l][p][w] = 1
                                psi_moment_down[position] = psi_down
                                psi_moment_up[position] = psi_down
                                H_effective[i][j][k][l][p][w] = matrix_term(H_list, psi_moment_down, psi_moment_up)
                                psi_down[i][j][k] = 0
                                psi_up[l][p][w] = 0

    return H_effective
