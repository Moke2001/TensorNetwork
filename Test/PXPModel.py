##  求PXP模型的基态和基态能量
import os
import numpy as np
from matlab import engine
from Basis.Operator.Hamiltonian.HamiltonianPXP import *
from Basis.Operator.Hamiltonian.HamiltonianTransverseFieldIsing1 import *
from DensityMatrixRenormalizationGroup.DMRG import dmrg


##  DMRG求PXP模型基态能量
def main():
    H_list=hamiltonian_pxp(5,[1]*5,[0.5]*5,200)
    psi, energy_result=dmrg(H_list, 10)
    print(energy_result)


##  DMRG求PXP模型基态能量
def main1():
    J_list=np.linspace(0,5,20)
    energy_ground=[]
    for i in range(len(J_list)):
        print(i)
        H_list=hamiltonian_transverse_field_ising1(7,[J_list[i]]*6,[1]*7)
        psi, energy_result=dmrg(H_list, 15)
        energy_ground.append(energy_result)
    eng=engine.start_matlab()
    energy_ground=np.array(energy_ground)
    eng.plot(J_list,energy_ground,'LineWidth',1.5)
    eng.grid('on',nargout=0)
    eng.xlabel('J',nargout=0)
    eng.ylabel('Energy of Ground State',nargout=0)
    os.system('pause')


if __name__ == '__main__':
    main1()