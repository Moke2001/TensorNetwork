##  为了方便两个MPS列表中的张量相乘设置的工具函数
import sys
import numpy as np


def mps_tensor_product(A,B):
    ##  数据预处理模块---------------------------------------------------------------------------------------------------------------------

    shape_A=A.shape  # 获得A的形状
    shape_B=B.shape  # 获得B的形状

    ## 核心算法模块------------------------------------------------------------------------------------------------------------------------

    ##  根据AB的形状选择缩并方式
    try:
        if len(shape_A)==1 and len(shape_B)==1:
            return np.einsum('i,i->', A, B)
        elif len(shape_A)==1 and len(shape_B)==2:
            return np.einsum('i,ij->j', A, B)
        elif len(shape_A)==2 and len(shape_B)==1:
            return np.einsum('ji,i->j', A, B)
        elif len(shape_A)==2 and len(shape_B)==2:
            return np.einsum('ij,jk->ik', A, B)
        elif len(shape_A)==1 and len(shape_B)==3:
            return np.einsum('i,ijk->jk', A, B)
        elif len(shape_A)==3 and len(shape_B)==1:
            return np.einsum('jki,i->jk', A, B)
        elif len(shape_A)==2 and len(shape_B)==3:
            return np.einsum('ij,jkl->ikl', A, B)
        elif len(shape_A)==3 and len(shape_B)==2:
            return np.einsum('ijk,kl->ijl', A, B)
        elif len(shape_A)==3 and len(shape_B)==3:
            return np.einsum('ijk,klp->ijlp', A, B)
        else:
            raise Exception('两张量形式不符合要求')

    ##  非MPS张量不允许调用该函数缩并
    except Exception as error:
        print(error)
        sys.exit()