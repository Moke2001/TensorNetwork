##  将算符作用在密度矩阵乘积态上
import numpy as np
from Basis.Operator.GeneralOperator.PauliOperator import pauli_operator
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.State.MatrixProductOperator import MatrixProductOperator
from Tool.SVD import svd_chi


def operate_operator(operator,rho_origin,chi,type):
    ##  参量类型检查模块-------------------------------------------------------------------------------------------------------------------

    assert isinstance(operator, Operator) or isinstance(operator, OperatorList),"operator必须是Operator类型或OperatorList类型"
    assert isinstance(rho_origin, MatrixProductOperator),"mps_list_origin必须是MatrixProductState类型"
    assert isinstance(chi,int),"chi必须是int类型"
    assert type == "L" or type =="R" or type =="l" or type =="r","type必须选择左右之一"

    rho = rho_origin.copy()  # 防止改变参数

    ##  单一算符计算模块------------------------------------------------------------------------------------------------------------------
    if type=="L" or type=="l":

        if isinstance(operator, Operator):

            ##  算符作用在一个位点的情况
            if isinstance(operator.target_index, int):
                rho.center_orthogonalization(operator.target_index)  # 在作用位点中心正交化

                ##  左端的作用方式
                if operator.target_index == 0:
                    rho.data[operator.target_index] = np.einsum('ab,bce->ace', operator.data,rho[operator.target_index])

                ##  右端的作用方式
                elif operator.target_index == rho.N - 1:
                    rho.data[operator.target_index] = np.einsum('ab,ebc->eac',operator.data, rho[operator.target_index])

                ##  中间的作用方式
                else:
                    rho.data[operator.target_index] = np.einsum('ab,ebcf->eacf', operator.data,rho[operator.target_index])

            ##  算符作用在两个位点的情况，要求两个位点相邻
            elif isinstance(operator.target_index, list):
                assert operator.target_index[0]==operator.target_index[1]-1,"作用的两个位点必须相邻"

                ##  提取两个作用位点的序号，并在这里中心正交化
                index_0 = operator.target_index[0]
                index_1 = operator.target_index[1]
                rho.center_orthogonalization(operator.target_index[0])

                ##  左端的作用方式
                if operator.target_index[0] == 0 and operator.target_index[1] != rho.N - 1:
                    moment = np.einsum('abc,cdef->adbef', rho[index_0], rho[index_1])
                    moment = np.einsum('ghad,adbef->ghbef', operator.data,moment)
                    shape=moment.shape
                    U, V = svd_chi(moment.reshape(shape[0]*shape[1], shape[2] * shape[3]*shape[4]), chi)
                    rho[index_0] = U.reshape(shape[0],shape[1],V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0], shape[2], shape[3],shape[4])

                ##  右端的作用方式
                elif operator.target_index[0] != 0 and operator.target_index[1] == rho.N - 1:
                    moment = np.einsum('abed,dcf->abcef', rho[index_0], rho[index_1])
                    moment = np.einsum('ghbc,abcef->aghef', operator.data,moment)
                    shape = moment.shape
                    U, V = svd_chi(moment.reshape(shape[0] * shape[1]*shape[2],shape[3]*shape[4]), chi)
                    rho[index_0] = U.reshape(shape[0], shape[1],shape[2], V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0],shape[3],shape[4])

                ##  两端的作用方式
                elif operator.target_index[0] == 0 and operator.target_index[1] == rho.N - 1:
                    moment = np.einsum('abc,cde->adbe', rho[index_0], rho[index_1])
                    moment = np.einsum('fgad,adbe->fgbe', operator.data,moment)
                    shape=moment.shape
                    U, V = svd_chi(moment.reshape(shape[0] *shape[1],shape[2]*shape[3]), chi)
                    rho[index_0] = U.reshape(shape[0],shape[1],V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0],shape[2],shape[3])

                ##  中间的作用方式
                else:
                    moment = np.einsum('fabc,cdeg->fadbeg', rho[index_0], rho[index_1])
                    moment = np.einsum('hiad,fadbeg->fhibeg', operator.data,moment)
                    shape=moment.shape
                    U, V = svd_chi(moment.reshape(shape[0] *shape[1]*shape[2],shape[3] * shape[4]*shape[5]), chi)
                    rho[index_0] = U.reshape(shape[0],shape[1],shape[2],V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0], shape[3],shape[4],shape[5])

    ##  单一算符计算模块------------------------------------------------------------------------------------------------------------------
    elif type == "R" or type == "r":

        if isinstance(operator, Operator):

            ##  算符作用在一个位点的情况
            if isinstance(operator.target_index, int):
                rho.center_orthogonalization(operator.target_index)  # 在作用位点中心正交化

                ##  左端的作用方式
                if operator.target_index == 0:
                    rho.data[operator.target_index] = np.einsum('bce,cd->bde',rho[operator.target_index], operator.data)

                ##  右端的作用方式
                elif operator.target_index == rho.N - 1:
                    rho.data[operator.target_index] = np.einsum('ebc,cd->ebd',rho[operator.target_index], operator.data)

                ##  中间的作用方式
                else:
                    rho.data[operator.target_index] = np.einsum('ebcf,cd->ebdf',rho[operator.target_index], operator.data)

            ##  算符作用在两个位点的情况，要求两个位点相邻
            elif isinstance(operator.target_index, list):
                assert operator.target_index[0] == operator.target_index[1] - 1, "作用的两个位点必须相邻"

                ##  提取两个作用位点的序号，并在这里中心正交化
                index_0 = operator.target_index[0]
                index_1 = operator.target_index[1]
                rho.center_orthogonalization(operator.target_index[0])

                ##  左端的作用方式
                if operator.target_index[0] == 0 and operator.target_index[1] != rho.N - 1:
                    moment = np.einsum('abc,cdef->adbef', rho[index_0], rho[index_1])
                    moment = np.einsum('adbef,beij->adijf', moment, operator.data)
                    shape = moment.shape
                    U, V = svd_chi(moment.reshape(shape[0] * shape[1], shape[2] * shape[3] * shape[4]), chi)
                    rho[index_0] = U.reshape(shape[0], shape[1], V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0], shape[2], shape[3], shape[4])

                ##  右端的作用方式
                elif operator.target_index[0] != 0 and operator.target_index[1] == rho.N - 1:
                    moment = np.einsum('abed,dcf->abcef', rho[index_0], rho[index_1])
                    moment = np.einsum('abcef,efij->abcij',moment, operator.data)
                    shape = moment.shape
                    U, V = svd_chi(moment.reshape(shape[0] * shape[1] * shape[2], shape[3] * shape[4]), chi)
                    rho[index_0] = U.reshape(shape[0], shape[1], shape[2], V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0], shape[3], shape[4])

                ##  两端的作用方式
                elif operator.target_index[0] == 0 and operator.target_index[1] == rho.N - 1:
                    moment = np.einsum('abc,cde->adbe', rho[index_0], rho[index_1])
                    moment = np.einsum('adbe,behi->adhi',moment, operator.data)
                    shape = moment.shape
                    U, V = svd_chi(moment.reshape(shape[0] * shape[1], shape[2] * shape[3]), chi)
                    rho[index_0] = U.reshape(shape[0], shape[1], V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0], shape[2], shape[3])

                ##  中间的作用方式
                else:
                    moment = np.einsum('fabc,cdeg->fadbeg', rho[index_0], rho[index_1])
                    moment = np.einsum('fadbeg,bejk->fadjkg',moment, operator.data)
                    shape = moment.shape
                    U, V = svd_chi(moment.reshape(shape[0] * shape[1] * shape[2], shape[3] * shape[4] * shape[5]), chi)
                    rho[index_0] = U.reshape(shape[0], shape[1], shape[2], V.shape[0])
                    rho[index_1] = V.reshape(V.shape[0], shape[3], shape[4], shape[5])

    ##  算符列表递归模块-------------------------------------------------------------------------------------------------------------------

    elif isinstance(operator, OperatorList):

        ##  单位点算符作用的循环
        for i in range(len(operator.single_list)):
            rho=operate_operator(operator.single_list[i], rho, chi,type)

        ##  双位点算符作用的循环
        for i in range(len(operator.double_list)):
            rho=operate_operator(operator.double_list[i], rho, chi,type)

    ##  结果处理模块-----------------------------------------------------------------------------------------------------------------------

    ##  将结果MPS重新中心正交化
    rho.center_orthogonalization(rho_origin.n_c)

    ##  返回结果
    return rho


if __name__ == '__main__':
    operator_test=pauli_operator(3,1,'z')
    rho_test=MatrixProductOperator.random_mpo([2,2,2],1,3)
    result_test=operate_operator(operator_test, rho_test, 3,'l')
    pass