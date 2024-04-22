##  定义多体系统的泡利算符
from qutip import *
import sys


def operator_sigma(N, j,arg):
    ##  参量检查模块-----------------------------------------------------------------------------------------------------------------------

    try:
        if arg=='x'or arg=='X':
            moment=sigmax()
        elif arg=='y'or arg=='Y':
            moment=sigmay()
        elif arg=='z'or arg=='Z':
            moment=sigmaz()
        elif arg=='i' or arg=='I':
            moment=identity(2)
        elif arg=='+':
            moment=(sigmax()+1j*sigmay())/2
        elif arg=='-':
            moment=(sigmax()-1j*sigmay())/2
        elif arg=='1':
            moment=0.5*(sigmaz()+identity(2))
        elif arg=='0':
            moment=0.5*(-sigmaz()+identity(2))
        else:
            raise ValueError('不存在这样的算符类型')

    ##  错误的输入提示并终止程序
    except ValueError as e:
        print(e)
        sys.exit()

    ##  核心计算模块-----------------------------------------------------------------------------------------------------------------------

    ##  如果是第零个位点则用Pauli算符做初始化
    if j == 0:
        result = moment
    else:
        result = identity(2)

    ##  循环，位点上用泡利算符，否则用单位算符
    for i in range(1, N):
        if i == j:
            result = tensor(result, moment)
        else:
            result = tensor(result, identity(2))

    ##  结果返回模块-----------------------------------------------------------------------------------------------------------------------

    ##  返回结果
    return result
