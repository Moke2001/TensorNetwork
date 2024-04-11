##  将算符作用在态矢上求结果
import numpy as np
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.State.MatrixProductState import MatrixProductState
from Tool.SVD import svd_chi


def operator_state(operator, mps_list_origin, chi):
    ##  参量类型检查模块-------------------------------------------------------------------------------------------------------------------

    assert isinstance(operator, Operator) or isinstance(operator, OperatorList),"operator必须是Operator类型或OperatorList类型"
    assert isinstance(mps_list_origin, MatrixProductState),"mps_list_origin必须是MatrixProductState类型"
    assert isinstance(chi,int),"chi必须是int类型"

    mps_list = mps_list_origin.copy()  # 防止改变参数

    ##  单一算符计算模块-------------------------------------------------------------------------------------------------------------------

    if isinstance(operator, Operator):

        ##  算符作用在一个位点的情况
        if isinstance(operator.target_index, int):
            mps_list.center_orthogonalization(operator.target_index)  # 在作用位点中心正交化

            ##  左端的作用方式
            if operator.target_index == 0:
                mps_list.data[operator.target_index] = np.einsum('ab,bc->ac', operator.data,mps_list[operator.target_index])

            ##  右端的作用方式
            elif operator.target_index == mps_list.N - 1:
                mps_list.data[operator.target_index] = np.einsum('cb,ab->ac',operator.data, mps_list[operator.target_index])

            ##  中间的作用方式
            else:
                mps_list.data[operator.target_index] = np.einsum('db,abc->adc', operator.data,mps_list[operator.target_index])

        ##  算符作用在两个位点的情况，要求两个位点相邻
        elif isinstance(operator.target_index, list):
            assert operator.target_index[0]==operator.target_index[1]-1,"作用的两个位点必须相邻"

            ##  提取两个作用位点的序号，并在这里中心正交化
            index_0 = operator.target_index[0]
            index_1 = operator.target_index[1]
            mps_list.center_orthogonalization(operator.target_index[0])

            ##  左端的作用方式
            if operator.target_index[0] == 0 and operator.target_index[1] != mps_list.N - 1:
                moment = np.einsum('ab,bcd->acd', mps_list[index_0], mps_list[index_1])
                moment = np.einsum('efac,acd->efd', operator.data,moment)
                U, V = svd_chi(moment.reshape(moment.shape[0], moment.shape[1] * moment.shape[2]), chi)
                mps_list[index_0] = U
                mps_list[index_1] = V.reshape(U.shape[1], moment.shape[1], moment.shape[2])

            ##  右端的作用方式
            elif operator.target_index[0] != 0 and operator.target_index[1] == mps_list.N - 1:
                moment = np.einsum('abc,cd->abd', mps_list[index_0], mps_list[index_1])
                moment = np.einsum('efbd,abd->aef', operator.data,moment)
                U, V = svd_chi(moment.reshape(moment.shape[0] * moment.shape[1], moment.shape[2]), chi)
                mps_list[index_0] = U.reshape(moment.shape[0], moment.shape[1], V.shape[0])
                mps_list[index_1] = V

            ##  两端的作用方式
            elif operator.target_index[0] == 0 and operator.target_index[1] == mps_list.N - 1:
                moment = np.einsum('ab,bc->ac', mps_list[index_0], mps_list[index_1])
                moment = np.einsum('cdab,ab->cd', operator.data,moment)
                U, V = svd_chi(moment.reshape(moment.shape[0] * moment.shape[1], moment.shape[2]), chi)
                mps_list[index_0] = U
                mps_list[index_1] = V

            ##  中间的作用方式
            else:
                moment = np.einsum('abc,cde->abde', mps_list[index_0], mps_list[index_1])
                moment = np.einsum('abcd,ecdf->eabf', operator.data,moment)
                U, V = svd_chi(moment.reshape(moment.shape[0] * moment.shape[1], moment.shape[2] * moment.shape[3]), chi)
                mps_list[index_0] = U.reshape(moment.shape[0], moment.shape[1], V.shape[0])
                mps_list[index_1] = V.reshape(U.shape[1], moment.shape[2], moment.shape[3])

    ##  算符列表递归模块-------------------------------------------------------------------------------------------------------------------

    elif isinstance(operator, OperatorList):

        ##  单位点算符作用的循环
        for i in range(len(operator.single_list)):
            mps_list=operator_state(operator.single_list[i], mps_list, chi)

        ##  双位点算符作用的循环
        for i in range(len(operator.double_list)):
            mps_list=operator_state(operator.double_list[i], mps_list, chi)

    ##  结果处理模块-----------------------------------------------------------------------------------------------------------------------

    ##  将结果MPS重新中心正交化
    mps_list.center_orthogonalization(mps_list_origin.n_c)

    ##  返回结果
    return mps_list
