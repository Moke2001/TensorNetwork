##  定义坍缩算符列表类
import copy
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

    ##  生成局域衰减类型Lindblad对象的静态函数-----------------------------------------------------------------------------------------

    @classmethod
    def local_decay(cls,N,gamma_list):
        ##  类型检查模块
        assert isinstance(gamma_list, list),"gamma_list必须是list类型"
        assert isinstance(N,int),"N必须是int类型"
        assert all(isinstance(term, int) or isinstance(term,float) for term in gamma_list),"gamma_list中的元素必须是int类型或float类型"
        assert len(gamma_list) == N,"N必须与gamma_list长度相等"

        single_list=[]  # 列表初始化

        ##  按照gamma_list中参数数据给出每个衰减形式的Lindblad算符
        for i in range(N):
            single_list.append(gamma_list[i]*Operator.operator_sigmaminus(N,i))

        ##  返回结果
        return LindbladList('decay',N,single_list)

    ##  复制函数----------------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return copy.deepcopy(self)

