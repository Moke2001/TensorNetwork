##  定义坍缩算符列表类
import copy
from Basis.Operator.GeneralOperator.PauliOperator import pauli_operator
from Basis.Operator.Operator import Operator


class LindbladList(object):
    ##  构造函数----------------------------------------------------------------------------------------------------------------------------

    def __init__(self, name, N, single_list):
        ##  类型判断
        assert isinstance(name, str),"name必须是string类型"
        assert isinstance(N,int),"N必须是int类型"
        assert isinstance(single_list, list),"single_list必须是list类型"
        assert all(isinstance(item, Operator) for item in single_list),"single_list内的元素必须是Operator类型"
        assert len(single_list) == N,"single_list内元素个数必须与局域个数对应"

        ##  赋予对象参数
        self.name = name  # 名称
        self.N = N  # 局域个数
        self.single_list = single_list  # 对应每个局域的单个坍缩算符

    ##  复制函数----------------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return copy.deepcopy(self)

