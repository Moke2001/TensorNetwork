##  基于量子轨迹和TEBD实现开放量子系统演化的模拟
import numpy as np
from Basis.Operator.LindbladList import LindbladList
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.Relation.Expect import expect
from Basis.Relation.OperateState import operator_state
from Basis.State.MatrixProductState import MatrixProductState
from TimeEvolveBlockDecimation.TEBD import tebd_state


def quantum_trajectory(psi, H_list, C_list, expect_operator, t_0, t_1, delta_t):
    ##  类型检查模块-----------------------------------------------------------------------------------------------------------------------

    assert isinstance(psi, MatrixProductState), "psi必须是MatrixProductState类型"
    assert isinstance(H_list, OperatorList), "H_list必须是OperatorList类型"
    assert isinstance(C_list, LindbladList), "C_list必须是LindbladList类型"
    assert isinstance(expect_operator,Operator), "expect_operator必须是Operator类型"
    assert isinstance(t_0,float) or isinstance(t_0,int),"t_0应是整数或浮点数"
    assert isinstance(t_1, float) or isinstance(t_1, int), "t_1应是整数或浮点数"
    assert isinstance(delta_t, float) or isinstance(delta_t, int), "delta_t应是整数或浮点数"

    ##  基本参数设置模块------------------------------------------------------------------------------------------------------------------

    num_sample = 2000  # 量子轨迹样本数目

    ##  参数预处理模块---------------------------------------------------------------------------------------------------------------------

    ##  构造非厄米演化的哈密顿量
    H_list_new = H_list.copy()  # 新的哈密顿量初始化
    for i in range(len(C_list.single_list)):
        nonhermitian_term = -(1j / 2) * C_list.single_list[i].dagger() * C_list.single_list[i]  # 非厄米项
        hermitian_term = H_list_new.single_list[i]  # 厄米项
        H_list_new.single_list[i] = hermitian_term + nonhermitian_term  # 非厄米哈密顿量

    ##  核心算法模块-----------------------------------------------------------------------------------------------------------------------

    expect_result = []  # 可观测量期望值时间序列结果初始化

    ##  表征选取大量样本模拟量子轨迹的循环
    for i in range(num_sample):
        print("当前进度："+str((i+1)*100/num_sample)+'%')  # 进度
        psi_moment = None  # 随时间变化的MPS初始化
        expect_value = []  # 本次量子轨迹可观测量期望值时间序列初始化

        ##  表征含时变化的循环
        for j in range(int((t_1 - t_0) / delta_t)):

            ##  首次演化用输入的psi进行计算
            if j == 0:
                expect_value.append(expect(expect_operator, psi))  # 将零时刻期望值输入
                psi_moment = tebd_state(psi, H_list_new, t_0 + j * delta_t, t_0 + (j + 1) * delta_t, delta_t / 5, 3)  # TEBD演化微分时间

                ##  首次演化的量子跳跃几率计算
                delta = 1 - (psi_moment * psi_moment).real  # 跳跃几率的计算
                delta_list = []  # 不同跳跃方向几率列表初始化

                ##  对不同的坍缩算符求跳跃几率
                for k in range(len(C_list.single_list)):
                    delta_list.append((delta_t * expect(C_list.single_list[k].dagger() * C_list.single_list[k], psi)).real)

                ##  随机量子跳跃
                np.random.seed()  # 初始化一个随机数
                if np.random.rand() < delta:
                    index = np.random.choice(np.array(range(len(C_list.single_list))), size=1,p=np.array(delta_list) / np.sum(delta_list))[0]
                    psi_moment = (delta_t / np.sqrt(delta_list[index])) * operator_state(C_list.single_list[index], psi,3)  # 跳跃并归一化
                else:
                    psi_moment = psi_moment / np.sqrt(1 - delta)  # 不跳跃并归一化

                ##  计算此刻可观测量的期望，并添加到本次的期望值时间序列中
                expect_value.append(expect(expect_operator, psi_moment))

            ##  非首次演化用psi_moment进行计算
            else:
                psi_pre = psi_moment.copy()  # 将此刻的MPS拷贝下来
                psi_moment = tebd_state(psi_pre, H_list_new, t_0 + j * delta_t, t_0 + (j + 1) * delta_t, delta_t / 5, 3)  # TEBD求微分时间演化

                ##  非首次演化的量子跳跃几率计算
                delta = 1 - (psi_moment * psi_moment).real
                delta_list = []

                ##  对不同的坍缩算符求跳跃几率
                for k in range(len(C_list.single_list)):
                    delta_list.append((delta_t * expect(C_list.single_list[k].dagger() * C_list.single_list[k], psi_pre)).real)

                ##  随机量子跳跃
                np.random.seed()  # 初始化一个随机数
                if np.random.rand() < delta:
                    index = np.random.choice(np.array(range(len(C_list.single_list))), size=1,p=np.array(delta_list) / np.sum(delta_list))[0]
                    psi_moment = np.sqrt(delta_t / delta_list[index]) * operator_state(C_list.single_list[index],psi_pre, 3)  # 跳跃并归一化
                else:
                    psi_moment = psi_moment / np.sqrt(1 - delta)  # 不跳跃并归一化

                ##  计算此刻可观测量的期望，并添加到本次的期望值时间序列中
                expect_value.append(expect(expect_operator, psi_moment))

        ##  判断是否是第一个样本，将所有样本期望对应点求和
        if i == 0:
            expect_result = np.array(expect_value)
        else:
            expect_result = expect_result + np.array(expect_value)

    ##  结果处理与输出模块----------------------------------------------------------------------------------------------------------------

    ##  对期望值时间序列样本取平均，得到最终结果
    expect_result = expect_result / num_sample

    ##  返回期望值的时间序列和对应的时间
    return expect_result,np.linspace(t_0,t_1,len(expect_result))
