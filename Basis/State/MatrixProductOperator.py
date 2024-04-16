##  矩阵乘积算符类
import copy
import numpy as np
from Tool.MPOTensorProduct import mpo_tensor_product
from Tool.SVD import svd_chi, svd_lr


class MatrixProductOperator(object):
    ##  构造函数----------------------------------------------------------------------------------------------------------------------------
    def __init__(self, N, data, n_c, chi):
        ##  参量检查模块
        assert isinstance(N, int), "N必须是int类型"
        assert isinstance(data, list), "data必须是list类型"
        assert isinstance(chi, int), "chi必须是int类型"
        assert all(isinstance(term, np.ndarray) for term in data), "data中的元素必须是np.ndarray类型"
        assert N == len(data), "N必须与data的长度相等"
        assert chi > 0, "chi必须大于0"

        ##  赋值模块
        self.N = N  # 局域个数
        self.data = data  # MPS数据
        self.n_c = n_c  # 中心序号
        self.chi = chi  # 截断维度，虚拟指标维度上界
        self.center_orthogonalization(n_c)

    ##  重载方括号取值函数----------------------------------------------------------------------------------------------------------------

    def __getitem__(self, index):
        ##  参量检查模块
        assert isinstance(index, int), "index必须是int类型"

        ##  取值
        return self.data[index]

    ##  重载方括号赋值函数----------------------------------------------------------------------------------------------------------------

    def __setitem__(self, index, value):
        ##  参量检查模块
        assert isinstance(index, int), "index必须是int类型"
        assert isinstance(value, np.ndarray), "value必须是np.ndarray类型"

        ##  赋值
        self.data[index] = value

    ##  重载左乘函数-----------------------------------------------------------------------------------------------------------------------

    def __mul__(self, Q):
        ##  参量检查模块
        Q_type = isinstance(Q, MatrixProductOperator) or isinstance(Q, int) or isinstance(Q, float) or isinstance(Q, complex)
        assert Q_type, "对方必须是MatrixProductState类型、int类型、float类型或complex类型"

        ##  MatrixProductState类型时做内积运算
        if isinstance(Q, MatrixProductOperator):

            ##  初始化
            start = np.einsum('abc,abd->cd', self.data[0], Q.data[0].conjugate())
            result = 0

            ##  将每个局域的张量缩并的循环
            for i in range(1, self.N):
                if i != self.N - 1:
                    start = np.einsum('ab,adcg,bdch->gh', start, self.data[i], Q.data[i].conjugate())
                else:
                    result = np.einsum('ab,adc,bdc->', start, self.data[i], Q.data[i].conjugate())

        ##  数值类型时做数乘运算
        else:
            data_new = []  # 结果数据初始化
            Q = Q ** (1 / self.N)  # 数值归一化

            ##  将归一化后的Q乘在每个MPS张量上
            for i in range(self.N):
                data_new.append(self.data[i].copy() * Q)

            ##  结果赋值并中心正交化
            result = MatrixProductOperator(self.N, data_new, self.n_c, self.chi)

        ##  返回结果
        return result

    ##  重载右乘函数-----------------------------------------------------------------------------------------------------------------------

    def __rmul__(self, Q):
        ##  参量检查模块
        Q_type = isinstance(Q, int) or isinstance(Q, float) or isinstance(Q, complex)
        assert Q_type, "对方必须是int类型、float类型或complex类型"

        ##  初始化
        data_new = []  # 结果数据初始化
        Q = Q ** (1 / self.N)  # 数值归一化

        ##  将归一化后的Q乘在每个MPS张量上
        for i in range(self.N):
            data_new.append(self.data[i].copy() * Q)

        ##  结果赋值并中心正交化
        result = MatrixProductOperator(self.N, data_new, self.n_c, self.chi)

        ##  返回结果
        return result

    ##  重载除法函数-----------------------------------------------------------------------------------------------------------------------

    def __truediv__(self, Q):
        ##  参量类型检查模块
        Q_type = isinstance(Q, int) or isinstance(Q, float)
        assert Q_type, "Q必须是一个浮点数或整数"

        ##  初始化
        data_new = []  # 结果数据初始化
        Q = Q ** (1 / self.N)  # 数值归一化

        ##  将归一化后的Q除在每个MPS张量上
        for i in range(self.N):
            data_new.append(self.data[i].copy() / Q)

        ##  结果赋值并中心正交化
        result = MatrixProductOperator(self.N, data_new, self.n_c, self.chi)

        ##  返回结果
        return result

    ##  将直积态改写为MPS态的静态函数--------------------------------------------------------------------------------------------------

    @classmethod
    def straight2mpo(cls, data):
        ##  参数检查模块
        assert isinstance(data, list), "data必须是list类型"
        assert all(isinstance(term, np.ndarray) for term in data), "data中的元素必须是np.ndarray类型"

        ##  计算模块
        mps_list = []  # 结果初始化

        ##  对于每个张量都让它长出虚拟指标
        for i in range(len(data)):

            ##  头结点只长右虚拟指标
            if i == 0:
                mps_list.append(data[i].reshape(data[i].shape[0],data[i].shape[1],1))

            ##  尾结点只长左虚拟指标
            elif i == len(data) - 1:
                mps_list.append(data[i].reshape(1, data[i].shape[0],data[i].shape[1]))

            ##  其余结点同时长左右两个虚拟指标
            else:
                mps_list.append(data[i].reshape(1, data[i].shape[0], data[i].shape[1],1))

        ## 将结果变成一个MPS类型对象，并中心正交化
        result = MatrixProductOperator(len(data), mps_list, 0, 1)

        ##  返回结果
        return result

    ##  将张量做低秩TT近似的静态函数----------------------------------------------------------------------------------------------------

    @classmethod
    def tensor2mpo(cls, T_origin, chi):
        ##  参量类型检查模块
        assert isinstance(T_origin, np.ndarray), "T_origin必须是np.ndarray类型"
        assert isinstance(chi, int), "chi必须是int类型"

        ##  初始化
        T = T_origin.copy()  # 防止改变参量
        num_qubit = len(T.shape)  # 张量的局域个数
        train_list = []  # 张量分解列表初始化

        ##  对张量做MPS分解
        for i in range(num_qubit):

            ##  如果是第一个，返回的是一个向量
            if i == 0:
                shape = T.shape  # 获得此刻T的形状
                dim_total = np.prod(T.shape)  # 得到此时的总维度
                T = T.reshape(shape[0],shape[1],int(dim_total /(shape[0]*shape[1])))  # 将张量后面的指标合并，形成一个矩阵
                U, V = svd_chi(T, chi)  # 对矩阵做奇异值分解并裁剪
                U.reshape(shape[0],shape[1],V.shape[0])
                train_list.append(U)  # 得到的U是一个向量，可以直接添加进去
                T = V.reshape((V.shape[0],).__add__(shape[2:]))  # 将V张量重构，第零个指标是虚拟指标

            ##  如果是最后一个，则返回已经成型的向量
            elif i == num_qubit - 1:
                train_list.append(T)

            ##  其他情况返回矩阵
            else:
                shape = T.shape  # 此刻张量的形状
                dim_total = np.prod(shape)  # 得到此时的总维度
                T = T.reshape(shape[0] * shape[1]*shape[2], int(dim_total / (shape[0] * shape[1]*shape[2])))  # 将张量后面的指标合并，虚拟指标和待分解指标合并
                U, V = svd_chi(T, chi)  # 对矩阵做奇异值分解并裁剪
                U = U.reshape(shape[0], shape[1],shape[2],U.shape[1])  # 将U重构为前虚拟指标、物理指标和后虚拟指标
                T = V.reshape((V.shape[0],).__add__(shape[3:]))  # 将张量重构，第零个指标是虚拟指标
                train_list.append(U)  # 将U添加进列表中

        ##  将结果做成一个MPS对象，并中心正交化
        result = MatrixProductOperator(num_qubit, train_list, 0, chi)

        ##  返回结果
        return result

    ##  将MPS改写为中心正交形式的成员函数----------------------------------------------------------------------------------------------

    def center_orthogonalization(self, n_c):
        ## 参量检查模块
        assert isinstance(n_c, int), "n_c必须是int类型"
        assert 0 <= n_c < self.N, "n_c不能超出范围"

        ##  将中心左侧的张量正交化
        for i in range(n_c):

            ##  对第零个张量单独处理
            if i == 0:
                shape = self.data[i].shape
                moment=self.data[i].copy()
                moment=moment.reshape(shape[0] * shape[1], shape[2])
                U, V = svd_lr(moment, 'l', self.chi)
                self.data[i] = U.reshape(shape[0],shape[1],V.shape[0])
                self.data[i + 1] = mpo_tensor_product(V, self.data[i + 1])

            ##  其余正常处理
            else:
                shape=self.data[i].shape
                moment = self.data[i].copy()
                moment = moment.reshape(shape[0] * moment.shape[1]*shape[2], moment.shape[3])
                U, V = svd_lr(moment, 'l', self.chi)
                self.data[i] = U.reshape(shape[0],shape[1],shape[2], V.shape[0])
                self.data[i + 1] = mpo_tensor_product(V, self.data[i + 1])

        ##  将中心右侧的张量正交化
        for i in range(len(self.data) - 1, n_c, -1):

            ##  对最后的张量单独处理
            if i == len(self.data) - 1:
                shape=self.data[i].shape
                moment=self.data[i].copy()
                moment=moment.reshape(shape[0],shape[1]*shape[2])
                U, V = svd_lr(moment, 'r', self.chi)
                self.data[i] = V.reshape(U.shape[1],shape[1],shape[2])
                self.data[i - 1] = mpo_tensor_product(self.data[i - 1], U)

            ##  其余正常处理
            else:
                shape=self.data[i].shape
                moment = self.data[i].copy()
                moment = moment.reshape(shape[0], moment.shape[1] * moment.shape[2]*shape[3])
                U, V = svd_lr(moment, 'r', self.chi)
                self.data[i] = V.reshape(U.shape[1],shape[1],shape[2],shape[3])
                self.data[i - 1] = mpo_tensor_product(self.data[i - 1], U)

        ##  将中心位置修改
        self.n_c = n_c

    ##  复制函数----------------------------------------------------------------------------------------------------------------------------

    def copy(self):
        return copy.deepcopy(self)

    ##  求迹的成员函数---------------------------------------------------------------------------------------------------------------------

    def trace(self):
        ##  对张量初始化
        start = 0

        ##  求对角矩阵
        for i in range(self.N):
            if i == 0:
                start = np.diagonal(self[i], 0, 0, 1)
                start=np.einsum('ij->ji', start)
            elif i == self.N - 1:
                moment = np.diagonal(self[i], 0, 1, 2)
                start = np.einsum('...a,ab->...b', start, moment)
            else:
                moment = np.diagonal(self[i], 0, 1, 2)
                moment=np.einsum('ijk->ikj', moment)
                start = np.einsum('...b,bcd->...cd', start, moment)

        ##  对角矩阵求和
        result = np.sum(start)

        ##  返回结果
        return result

    @classmethod
    ##  随机产生一个MPS的静态函数-------------------------------------------------------------------------------------------------------
    def random_mpo(cls, dim_list, n_c, chi):
        ##  参量检查模块
        assert isinstance(dim_list, list) or isinstance(dim_list, np.ndarray), "dim_list格式必须是list类型或np.adarray类型"
        if isinstance(dim_list, np.ndarray):
            assert len(dim_list.shape) == 1, "dim_list必须是一维的"
        else:
            assert all(isinstance(term, int) for term in dim_list), "dim_list中所有元素都需要是int类型"
        assert isinstance(n_c, int), "n_c必须是int类型"
        assert isinstance(chi, int), "chi必须是int类型"

        ##  赋予每个MPS位点一个特定格式张量
        mpo_list = []  # 结果数据初始化
        for i in range(len(dim_list)):

            ##  头结点的处理
            if i == 0:
                moment = np.random.rand(int(dim_list[i]),int(dim_list[i]),chi)
                mpo_list.append(moment)

            ##  尾节点的处理
            elif i == len(dim_list) - 1:
                moment = np.random.rand(chi, int(dim_list[i]),int(dim_list[i]))
                mpo_list.append(moment)

            ## 中间结点的处理
            else:
                moment = np.random.rand(chi, int(dim_list[i]), int(dim_list[i]),chi)
                mpo_list.append(moment)

        ##  构造MPS对象并中心正交化、归一化
        result = MatrixProductOperator(len(dim_list), mpo_list, n_c, chi)
        result = result / np.abs(result.trace())

        ##  返回结果
        return result


if __name__ == '__main__':
    x=MatrixProductOperator.random_mpo([2,2,2,2],2,3)
    print(x.trace())