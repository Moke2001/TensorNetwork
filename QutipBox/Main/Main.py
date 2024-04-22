##  主函数，在这里可以调用各种算法、对象完成计算
import numpy as np
from QutipBox.Main.HamiltonianCreator import hamiltonian_creator
from matlab import engine
import os
from QutipBox.Operator.Hamiltonian.HamiltonianTransverseIsing1 import hamiltonian_transverse_ising1


##  验证V的大小对系统低能能谱的影响较小
def main_0():
    V_list=np.linspace(10,200,30)
    eng = engine.start_matlab()
    for i in range(3,8):
        phi = (1 + np.sqrt(5)) / 2   # 计算黄金分割率
        dim = int((phi**(i+1) - (-1/phi)**(i+1))/np.sqrt(5))
        E_matrix=np.zeros((dim,len(V_list)))
        print(i)
        for j in range(len(V_list)):
            H=hamiltonian_creator(i,[1]*i,[0.5]*i,V_list[j])
            eigenvalues, eigenstates = H.eigenstates()
            E_matrix[:,j]=eigenvalues[:dim]
        for j in range(dim):
            x=eng.plot(V_list,E_matrix[j][:],'LineWidth',1.5)
            eng.hold('on',nargout=0)
        eng.xlabel('V',nargout=0)
        eng.ylabel('Energy of Eigenstate',nargout=0)
        eng.grid('on',nargout=0)
        eng.saveas(x,str(i)+'x.pdf',nargout=0)
        eng.clf(nargout=0)


##  求系统的相图
def main_1():
    H=hamiltonian_transverse_ising1(8,[0.5]*7,[1]*8)
    e,s=H.eigenstates()
    print(e[0])


def main_2():
    t=np.linspace(0,10,1000)
    p=t.copy()
    for i in range(len(t)):
        r=np.mod(i,50)*(10/1000)
        if np.mod(i,50)==0:
            p[i]=1
        else:
            p[i]=1-np.sin(r*1)**2
    eng = engine.start_matlab()
    eng.plot(t, p, 'LineWidth', 1.5)
    eng.grid('on', nargout=0)
    eng.xlabel('J', nargout=0)
    eng.ylabel('Energy of Ground State', nargout=0)
    os.system('pause')


if __name__ == '__main__':
    main_2()