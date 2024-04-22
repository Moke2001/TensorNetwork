import sys
import numpy as np
from Basis.Operator.Operator import Operator


def pauli_operator(N,target_index,type):
    ##  类型检查模块
    assert isinstance(N, int), "N必须是int类型"
    assert isinstance(target_index,int),"target_index必须是int类型"
    assert isinstance(type,str),"type必须是string类型"

    ##  选择合适的核心数据
    try:
        if type == "X"or type == "x":
            data=np.array([[0,1],[1,0]])
        elif type == "Y"or type == "y":
            data=np.array([[0,-1j],[1j,0]])
        elif type == "Z"or type == "z":
            data = np.array([[1, 0], [0, -1]])
        elif type == "-":
            data = np.array([[0,0],[1,0]])
        elif type == "+":
            data = np.array([[0,1],[0,0]])
        elif type == "I"or type == "i":
            data = np.identity(2)
        elif type == "1":
            data = np.array([[1,0],[0,0]])
        elif type == "-1":
            data = np.array([[0,0],[0,1]])
        else:
            raise Exception("没有这样的Pauli算符类型")

    ##  报错模块
    except Exception as error:
        print(error)
        sys.exit()

    ##  返回结果
    return Operator('identity', N, target_index, data,2)