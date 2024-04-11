##  算符类的定义
import numpy as np
import copy


class Operator(object):
    ##  构造函数----------------------------------------------------------------------------------------------------------------------------

    def __init__(self,name,N,target_index,data,dim):
        ##  类型检查模块
        assert isinstance(name,str),"name必须是string类型"
        assert isinstance(N,int),"N必须是int类型"
        assert isinstance(target_index,int) or isinstance(target_index,list),"target必须是int类型或list类型"
        assert isinstance(data,np.ndarray),"data必须是np.ndarray类型"
        assert isinstance(dim,int) or isinstance(dim,list),"dim必须是int类型或list类型"

        ##  参数赋予模块
        self.name = name  # 名称
        self.N = N  # 局域个数
        self.target_index = target_index  # 作用的局域序号
        self.data = data  # 数据
        self.dim = dim  # 维度

    ##  重载右乘法函数---------------------------------------------------------------------------------------------------------------------

    def __rmul__(self,Q):
        ##  检查参量类型
        Q_type=isinstance(Q,float) or isinstance(Q,int) or isinstance(Q,complex)
        assert Q_type,"只能与数值进行乘法"

        ##  计算模块
        return Operator(self.name, self.N, self.target_index, Q*self.data,self.dim)

    ##  重载左乘法函数---------------------------------------------------------------------------------------------------------------------

    def __mul__(self, Q):
        ##  检查参量类型
        Q_type = isinstance(Q, Operator) or isinstance(Q, float) or isinstance(Q, int) or isinstance(Q, complex)
        assert Q_type, "Q必须是Operator类型、浮点数、整数或复数"

        ##  对方是算符的情况
        if isinstance(Q,Operator):

            ##  数据预处理
            data_0=self.data.copy()
            data_1=Q.data.copy()
            dim_0=self.dim
            dim_1=Q.dim
            target_index_0=self.target_index
            target_index_1 = Q.target_index

            ##  如果不在同一位点要张量积单位张量
            if target_index_0<target_index_1:
                moment_0=np.einsum('ia,jb->ijab',data_0,np.identity(dim_1))
                moment_1=np.einsum('ia,jb->ijab',np.identity(dim_0),data_1)
                data=np.einsum('ijab,abcd->ijcd',moment_0,moment_1)
                target_index = [target_index_0, target_index_1]
                dim = [dim_0, dim_1]

            ##  如果不在同一位点要张量积单位张量
            elif target_index_0>target_index_1:
                moment_0 = np.einsum('ia,jb->jiba', data_0, np.identity(dim_1))
                moment_1 = np.einsum('ia,jb->jiba', np.identity(dim_0), data_1)
                data = np.einsum('jiba,bacd->jicd', moment_0, moment_1)
                target_index = [target_index_1, target_index_0]
                dim = [dim_1, dim_0]

            ##  在同一位点直接矩阵乘法
            else:
                data=data_0@data_1
                target_index = target_index_0
                dim = dim_0

            ##  给结果赋值
            result=Operator('mul',self.N,target_index,data,dim)

        ##  对方是数的情况
        else:
            result=Operator('mul',self.N,self.target_index,Q*self.data,self.dim)

        return result

    ##  重载加法函数-----------------------------------------------------------------------------------------------------------------------

    def __add__(self, Q):
        ##  参量类型检查模块
        Q_type = isinstance(Q, Operator)
        assert Q_type, "Q必须是Operator类型"

        ##  计算模块
        data_new=self.data+Q.data
        return Operator('add',self.N,self.target_index,data_new,self.dim)

    ##  重载减法函数-----------------------------------------------------------------------------------------------------------------------

    def __sub__(self, Q):
        ##  检查参量类型
        Q_type = isinstance(Q, Operator)
        assert Q_type, "Q必须是Operator类型"

        ##  计算模块
        data_new = self.data - Q.data
        return Operator('add', self.N, self.target_index, data_new, self.dim)

    ##  对自身取厄米共轭的成员函数-------------------------------------------------------------------------------------------------------

    def dagger(self):
        ##  数据预处理模块
        data=self.data.copy()
        data=data.conjugate()

        ##  计算模块，讨论单体算符和两体算符
        if isinstance(self.target_index,int):
            data=np.einsum('ij->ji',data)
        else:
            data=np.einsum('ijab->abij',data)

        ##  返回结果
        return Operator('dagger',self.N,self.target_index,data,self.dim)

    ##  复制函数----------------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return copy.deepcopy(self)
