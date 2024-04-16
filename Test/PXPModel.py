##  求PXP模型的基态和基态能量
from Basis.Operator.Hamiltonian.HamiltonianPXP import hamiltonian_pxp
from DensityMatrixRenormalizationGroup.DMRG import dmrg


##  DMRG求PXP模型基态能量
def main():
    H_list=hamiltonian_pxp(5,[1]*5,[0.5]*5,200)
    psi, energy_result=dmrg(H_list, 10)
    print(energy_result)


if __name__ == '__main__':
    main()