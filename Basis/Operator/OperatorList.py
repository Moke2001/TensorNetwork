##  定义算符列表类，用于储存加和关系的各个算符
import scipy
import copy
from Basis.Operator.Operator import Operator


class OperatorList():
    ##  构造函数----------------------------------------------------------------------------------------------------------------------------

    def __init__(self,name,N,single_list,double_list):
        ##  类型检查模块
        assert isinstance(single_list,list), "single_list必须是list类型"
        assert isinstance(double_list,list),"double_list必须是list类型"
        assert isinstance(N,int),"N必须是int类型"
        assert isinstance(name,str),"name必须是string类型"
        assert all(isinstance(term,Operator) for term in single_list),"single_list中的元素必须是Operator类型"
        assert all(isinstance(term, Operator) for term in double_list),"double_list中的元素必须是Operator类型"
        assert N==len(single_list) and (N==len(double_list)+1 or N==len(double_list)),"N必须与参数列表相对应"

        ##  赋值模块
        self.name = name  # 名称
        self.N=N  # 局域个数
        self.single_list=single_list  # 单位点算符列表
        self.double_list=double_list  # 双位点算符列表

    ##  构造一维横场Ising模型的哈密顿量的静态函数--------------------------------------------------------------------------------------

    @classmethod
    def hamiltonian_transverse_ising1(cls, N, J_list, h_list):
        ##  类型检查模块
        assert isinstance(N,int),"N必须是int类型"
        assert isinstance(J_list,list),"J_list必须是list类型"
        assert isinstance(h_list,list),"h_list必须是list类型"
        assert N==len(J_list)+1 and N==len(h_list),"N必须与参数列表相对应"
        assert all(isinstance(term, float) or isinstance(term, int) for term in h_list), "single_list中的元素必须是int类型或folat类型"
        assert all(isinstance(term, float) or isinstance(term, int) for term in J_list), "double_list中的元素必须是int类型或folat类型"

        ##  算符列表初始化
        h_list_single = []
        h_list_double = []

        ##  将算符代入到每一个点位上
        for i in range(N - 1):
            if i != N - 2:
                sigmaz_0 = Operator.operator_sigmaz(N, i)
                sigmaz_1 = Operator.operator_sigmaz(N, i + 1)
                sigmax = Operator.operator_sigmax(N, i)
                h_list_single.append(h_list[i] * sigmax)
                h_list_double.append(J_list[i] * sigmaz_0 * sigmaz_1)
            else:
                sigmaz_0 = Operator.operator_sigmaz(N, i)
                sigmaz_1 = Operator.operator_sigmaz(N, i + 1)
                sigmax_0 = Operator.operator_sigmax(N, i)
                sigmax_1 = Operator.operator_sigmax(N, i + 1)
                h_list_single.append(h_list[i] * sigmax_0)
                h_list_single.append(h_list[i + 1] * sigmax_1)
                h_list_double.append(J_list[i] * sigmaz_0 * sigmaz_1)

        ##  返回结果
        result=OperatorList('TransverseFieldIsing1',N,h_list_single,h_list_double)
        return result

    ##  将哈密顿量转化为时间演化算符列表的静态函数-------------------------------------------------------------------------------------

    def evolve_operator(self,delta_t):
        ##  类型检查模块
        assert isinstance(delta_t,int) or isinstance(delta_t,float),"delta_t必须是int类型或float类型"

        ##  时间演化算符列表初始化
        U_list_single=[]
        U_list_double=[]

        ##  对每个哈密顿量分量求对应的时间演化算符
        for i in range(len(self.single_list)):
            moment=self.single_list[i].data.copy()
            moment=scipy.linalg.expm(-1j*moment*delta_t)  # 求演化算符
            U_list_single.append(Operator('Evolve',self.single_list[i].N,self.single_list[i].target_index,moment,self.single_list[i].dim))  # 将演化算符张量化

        for i in range(len(self.double_list)):
            moment = self.double_list[i].data.copy()
            shape=moment.shape
            moment=moment.reshape(shape[0]*shape[1],shape[2]*shape[3])
            moment = scipy.linalg.expm(-1j * moment * delta_t)  # 求演化算符
            moment = moment.reshape(shape[0],shape[1],shape[2],shape[3])
            U_list_double.append(Operator('Evolve',self.double_list[i].N,self.double_list[i].target_index,moment,self.double_list[i].dim))  # 将演化算符张量化

        ##  返回结果
        return OperatorList('TransverseFieldIsing1TimeEvolve',self.N,U_list_single,U_list_double)

    ##  复制函数----------------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return OperatorList(copy.deepcopy(self.name),copy.deepcopy(self.N),copy.deepcopy(self.single_list),copy.deepcopy(self.double_list))

