##  主函数，在这里可以调用各种算法、对象完成计算
from Basis.Relation.Expect import expect
from DensityMatrixRenormalizationGroup.DMRG import dmrg
from Main.HamiltonianCreator import hamiltonian_creator


def main():
    H_list=hamiltonian_creator(5,[1]*5,[0.5]*5,200)
    psi, energy_result=dmrg(H_list, 10)
    print(energy_result)


if __name__ == '__main__':
    main()