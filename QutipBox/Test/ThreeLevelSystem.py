import numpy as np
from matlab import engine
from qutip import *
import os

def main():
    t_list = np.linspace(0, 10, 1000)  # 时间区间
    psi_0=basis(3,0)
    psi_1=basis(3,1)
    psi_2=basis(3,2)
    H=psi_0*psi_1.dag()+psi_1*psi_0.dag()
    C=[psi_2*psi_1.dag()]
    E=[psi_0*psi_0.dag(),psi_1*psi_1.dag(),psi_2*psi_2.dag()]
    result = mesolve(H, psi_0, t_list, C, E).expect  # 结果

    ##  绘制图形
    eng = engine.start_matlab()
    eng.plot(t_list, result[0], 'LineWidth', 1.5)
    eng.hold('on', nargout=0)
    eng.plot(t_list, result[1], 'LineWidth', 1.5)
    eng.hold('on', nargout=0)
    eng.plot(t_list, result[2], 'LineWidth', 1.5)
    eng.hold('on', nargout=0)
    eng.xlabel('Time', nargout=0)
    eng.ylabel('Number of Possessions', nargout=0)
    eng.legend(nargout=0)
    eng.grid('on', nargout=0)
    os.system('pause')


if __name__=="__main__":
    main()