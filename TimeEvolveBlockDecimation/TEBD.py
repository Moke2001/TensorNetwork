##  TEBD算法用于求解量子系统态矢的演化
import numpy as np
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.Relation.Expect import expect
from Basis.Relation.OperateState import operate_state
from Basis.State.MatrixProductState import MatrixProductState


##  求某可观测量的含时变化时间序列
def tebd(psi, H_list, expect_operator, t_0, t_1, delta_t, chi):
    ##  类型检查模块-----------------------------------------------------------------------------------------------------------------------

    assert isinstance(psi, MatrixProductState), "psi应是MatrixProductState类型"
    assert isinstance(H_list, OperatorList), "H_list应是OperatorList类型"
    assert isinstance(expect_operator, Operator) or isinstance(expect_operator,OperatorList), "expect_operator应是Operator类型或OperatorList类型"
    assert isinstance(t_0, float) or isinstance(t_0, int), "t_0应是整数或浮点数"
    assert isinstance(t_1, float) or isinstance(t_1, int), "t_1应是整数或浮点数"
    assert isinstance(delta_t, float) or isinstance(delta_t, int), "delta_t应是整数或浮点数"

    ##  核心算法模块-----------------------------------------------------------------------------------------------------------------------

    psi_moment = psi.copy()  # 防止改变输入的参量
    expect_value = []  # 可观测量时间序列初始化
    U_list = H_list.evolve_operator(delta_t)  # 计算哈密顿量对应的时间演化算符列表

    ##  将时间演化算符作用在态矢上的循环并得到当前时刻的期望
    for i in range(int((t_1 - t_0) / delta_t)):
        print("当前进度：" + str((i + 1) * 100 / int((t_1 - t_0) / delta_t)) + '%')  # 进度
        psi_moment = operate_state(U_list, psi_moment, chi)
        expect_value.append(expect(expect_operator, psi_moment))

    ##  结果处理与输出模块----------------------------------------------------------------------------------------------------------------

    ##  返回结果
    return np.array(expect_value), np.linspace(t_0, t_1, len(expect_value))


##  求某态矢演化后的MPS
def tebd_state(psi, H_list, t_0, t_1, delta_t, chi):
    ##  类型检查模块-----------------------------------------------------------------------------------------------------------------------

    assert isinstance(psi, MatrixProductState), "psi应是MatrixProductState类型"
    assert isinstance(H_list, OperatorList), "H_list应是OperatorList类型"
    assert isinstance(t_0, float) or isinstance(t_0, int), "t_0应是整数或浮点数"
    assert isinstance(t_1, float) or isinstance(t_1, int), "t_1应是整数或浮点数"
    assert isinstance(delta_t, float) or isinstance(delta_t, int), "delta_t应是整数或浮点数"

    ##  核心算法模块-----------------------------------------------------------------------------------------------------------------------

    psi_moment = psi.copy()  # 防止改变输入的参量
    U_list = H_list.evolve_operator(delta_t)  # 可观测量时间序列初始化

    ##  将时间演化算符作用在态矢上的循环
    for i in range(int((t_1 - t_0) / delta_t)):
        psi_moment = operate_state(U_list, psi_moment, chi)

    ##  结果处理与输出模块----------------------------------------------------------------------------------------------------------------

    ##  返回结果
    return psi_moment
