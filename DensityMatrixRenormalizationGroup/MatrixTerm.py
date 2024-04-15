##  计算矩阵在两个MPS下的矩阵元
import numpy as np
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.State.MatrixProductState import MatrixProductState


def matrix_term(operator,mps_list_0_origin,mps_list_1_origin):
    ##  参量类型检查模块-------------------------------------------------------------------------------------------------------------------

    assert isinstance(operator,Operator) or isinstance(operator,OperatorList),"operator必须是Operator类型或OperatorList类型"
    assert isinstance(mps_list_0_origin,MatrixProductState),"mps_list_0必须是MatrixProductState类型"
    assert isinstance(mps_list_1_origin, MatrixProductState), "mps_list_1必须是MatrixProductState类型"

    ##  初始化模块--------------------------------------------------------------------------------------------------------------------------

    mps_list_0 = mps_list_0_origin.copy()  # 防止改变参量
    mps_list_1 = mps_list_1_origin.copy()  # 防止改变参量
    n_c=mps_list_0.n_c  # 中心位点
    N=mps_list_0.N  # 总位点数
    result=0  # 结果初始化

    ##  单一算符计算模块-------------------------------------------------------------------------------------------------------------------

    if isinstance(operator,Operator):

        ##  单个位点作用的情况
        if isinstance(operator.target_index,int):
            index_op = operator.target_index  # 算符作用位点
            index_c=n_c  # 中心位点

            ##  当作用在中心右边时
            if index_op>index_c:

                ##  作用在右端，中心在左端
                if index_op==N-1 and index_c==0:
                    list_end=np.einsum('ab,db,ed->ae',mps_list_0[index_op],operator.data,mps_list_1[index_op].conjugate())
                    list_start=np.einsum('ab,ac->bc',mps_list_0[index_c],mps_list_1[index_c].conjugate())
                    for i in range(index_c+1,index_op):
                        list_start=np.einsum('ab,acd,bce->de',list_start,mps_list_0[i],mps_list_1[i].conjugate())
                    result=np.einsum('ab,ab->',list_start,list_end)

                ##  作用在右端，中心不在左端
                elif index_op==N-1 and index_c!=0:
                    list_end = np.einsum('ab,db,ed->ae', mps_list_0[index_op], operator.data,mps_list_1[index_op].conjugate())
                    list_start = np.einsum('abc,abd->cd', mps_list_0[index_c], mps_list_1[index_c].conjugate())
                    for i in range(index_c + 1, index_op):
                        list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                    result = np.einsum('ab,ab->', list_start, list_end)

                ##  作用不在右端，中心在左端
                elif index_op!=N-1 and index_c==0:
                    list_end = np.einsum('abc,db,edc->ae', mps_list_0[index_op], operator.data,mps_list_1[index_op].conjugate())
                    list_start=np.einsum('ab,ac->bc',mps_list_0[index_c],mps_list_1[index_c].conjugate())
                    for i in range(index_c + 1, index_op):
                        list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                    result = np.einsum('ab,ab->', list_start, list_end)

                ##  作用不在右端，中心不在左端
                else:
                    list_end = np.einsum('abc,db,edc->ae', mps_list_0[index_op], operator.data,mps_list_1[index_op].conjugate())
                    list_start = np.einsum('abc,abd->cd', mps_list_0[index_c], mps_list_1[index_c].conjugate())
                    for i in range(index_c + 1, index_op):
                        list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                    result = np.einsum('ab,ab->', list_start, list_end)

            ##  当作用在中心左边时
            elif index_op<index_c:

                ##  作用在左端，中心在右端
                if index_c==N-1 and index_op==0:
                    list_start=np.einsum('ab,ca,cd->bd',mps_list_0[index_op],operator.data,mps_list_1[index_op].conjugate())
                    list_end=np.einsum('ab,cb->ac',mps_list_0[index_c],mps_list_1[index_c].conjugate())
                    for i in range(index_op+1,index_c):
                        list_start=np.einsum('ab,acd,bce->de',list_start,mps_list_0[i],mps_list_1[i].conjugate())
                    result=np.einsum('ab,ab->',list_start,list_end)

                ##  作用不在左端，中心在右端
                elif index_c==N-1 and index_op!=0:
                    list_start = np.einsum('abc,eb,aef->cf', mps_list_0[index_op], operator.data,mps_list_1[index_op].conjugate())
                    list_end = np.einsum('ab,cb->ac', mps_list_0[index_c], mps_list_1[index_c].conjugate())
                    for i in range(index_op + 1, index_c):
                        list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                    result = np.einsum('ab,ab->', list_start, list_end)

                ##  作用在左端，中心不在右端
                elif index_c!=N-1 and index_op==0:
                    list_start = np.einsum('ab,ca,cd->bd', mps_list_0[index_op], operator.data,mps_list_1[index_op].conjugate())
                    list_end=np.einsum('abc,dbc->ad',mps_list_0[index_c],mps_list_1[index_c].conjugate())
                    for i in range(index_op + 1, index_c):
                        list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                    result = np.einsum('ab,ab->', list_start, list_end)

                ##  作用不在左端，中心不在右端
                else:
                    list_start = np.einsum('abc,eb,aef->cf', mps_list_0[index_op], operator.data,mps_list_1[index_op].conjugate())
                    list_end = np.einsum('abc,dbc->ad', mps_list_0[index_c], mps_list_1[index_c].conjugate())
                    for i in range(index_op + 1, index_c):
                        list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                    result = np.einsum('ab,ab->', list_start, list_end)

            ##  作用与中心重合时
            else:

                ##  在左端
                if index_c==0:
                    result=np.einsum('bc,db,dc->',mps_list_0[index_op], operator.data, mps_list_1[index_op].conjugate())

                ##  在右端
                elif  index_c==N-1:
                    result = np.einsum('ab,db,ad->', mps_list_0[index_op], operator.data, mps_list_1[index_op].conjugate())

                ##  在中间
                else:
                    result = np.einsum('abc,db,adc->', mps_list_0[index_op], operator.data, mps_list_1[index_op].conjugate())

        ##  双位点作用情况
        elif isinstance(operator.target_index,list):
            index_0=operator.target_index[0]
            index_1=operator.target_index[1]
            index_n=n_c

            ##  两者重合时
            if index_0 == index_n or index_1 == index_n:

                ##  在两端
                if index_1 == N-1 and index_0==0:
                    s='bc,cd,bdfg,fh,hg->'

                ##  在左端
                elif index_1 != N-1 and index_0==0:
                    s = 'ab,bcd,egac,ef,fgd->'

                ##  在右端
                elif index_1 == N-1 and index_0!=0:
                    s = 'abc,cd,egbd,aef,fg->'

                ##  在中间
                else:
                    s = 'abc,cde,bdfg,afh,hge->'

                ##  求对应的结果
                result=np.einsum(s,mps_list_0[index_0],mps_list_0[index_1],operator.data,mps_list_1[index_0].conjugate(),mps_list_1[index_1].conjugate())

            ##  两者不重合时
            else:

                ##  作用在中心右端
                if index_0>index_n:

                    ##  作用在右端，中心在左端
                    if index_1==N-1 and index_n==0:
                        list_start=np.einsum('ab,ac->bc',mps_list_0[index_n],mps_list_1[index_n].conjugate())
                        s='abc,cd,gebd,hgf,fe->ah'
                        list_end=np.einsum(s,mps_list_0[index_0],mps_list_0[index_1],operator.data,mps_list_1[index_0].conjugate(),mps_list_1[index_1].conjugate())
                        for i in range(index_n+1,index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

                    ##  作用不在右端，中心在左端
                    elif index_1!=N-1 and index_n==0:
                        list_start = np.einsum('ab,ac->bc', mps_list_0[index_n], mps_list_1[index_n].conjugate())
                        s = 'abc,cdi,gebd,hgf,fei->ah'
                        list_end = np.einsum(s, mps_list_0[index_0], mps_list_0[index_1], operator.data,mps_list_1[index_0].conjugate(), mps_list_1[index_1].conjugate())
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

                    ##  作用在右端，中心不在左端
                    elif index_1 == N - 1 and index_n != 0:
                        list_start = np.einsum('abc,abd->cd', mps_list_0[index_n], mps_list_1[index_n].conjugate())
                        s = 'abc,cd,gebd,hgf,fe->ah'
                        list_end = np.einsum(s, mps_list_0[index_0], mps_list_0[index_1], operator.data,mps_list_1[index_0].conjugate(), mps_list_1[index_1].conjugate())
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

                    ##  作用不在右端，中心不在左端
                    else:
                        list_start = np.einsum('abc,abd->cd', mps_list_0[index_n], mps_list_1[index_n].conjugate())
                        s = 'abc,cdi,gebd,hgf,fei->ah'
                        list_end = np.einsum(s, mps_list_0[index_0], mps_list_0[index_1], operator.data,mps_list_1[index_0].conjugate(), mps_list_1[index_1].conjugate())
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

                ##  作用在中心左端时
                else:

                    ##  作用在左端，中心在右端
                    if index_n==N-1 and index_0==0:
                        list_end=np.einsum('ab,cb->ac',mps_list_0[index_n],mps_list_1[index_n].conjugate())
                        s='ab,bcd,efac,eg,gfh->dh'
                        list_start=np.einsum(s,mps_list_0[index_0],mps_list_0[index_1],operator.data,mps_list_1[index_0].conjugate(),mps_list_1[index_1].conjugate())
                        for i in range(index_n+1,index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

                    ##  作用在左端，中心不在右端
                    elif index_n!=N-1 and index_0==0:
                        list_end = np.einsum('abc,dbc->ad', mps_list_0[index_n], mps_list_1[index_n].conjugate())
                        s = 'ab,bcd,egac,ef,fgh->dh'
                        list_start = np.einsum(s, mps_list_0[index_0], mps_list_0[index_1], operator.data,mps_list_1[index_0].conjugate(), mps_list_1[index_1].conjugate())
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

                    ##  作用不在左端，中心在右端
                    elif index_n == N - 1 and index_0 != 0:
                        list_end = np.einsum('ab,cb->ac', mps_list_0[index_n], mps_list_0[index_n].conjugate())
                        s = 'iab,bcd,efac,ieg,gfh->dh'
                        list_start = np.einsum(s, mps_list_0[index_0], mps_list_0[index_1], operator.data,mps_list_1[index_0].conjugate(), mps_list_1[index_1].conjugate())
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

                    ##  作用不在左端，中心不在右端
                    else:
                        list_end = np.einsum('abc,dbc->ad', mps_list_0[index_n], mps_list_1[index_n].conjugate())
                        s = 'iab,bcd,efac,ieg,gfh->dh'
                        list_start = np.einsum(s, mps_list_0[index_0], mps_list_0[index_1], operator.data,mps_list_1[index_0].conjugate(), mps_list_1[index_1].conjugate())
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('ab,acd,bce->de', list_start, mps_list_0[i], mps_list_1[i].conjugate())
                        result = np.einsum('ab,ab->', list_start, list_end)

    ##  算符列表递归模块-------------------------------------------------------------------------------------------------------------------

    elif isinstance(operator,OperatorList):

        ##  单位点算符期望值的循环
        for i in range(len(operator.single_list)):
            result=result+matrix_term(operator.single_list[i],mps_list_0,mps_list_1)

        ##  双位点算符期望值的循环
        for i in range(len(operator.double_list)):
            result = result + matrix_term(operator.double_list[i], mps_list_0,mps_list_1)

    ##  结果返回模块-----------------------------------------------------------------------------------------------------------------------

    ##  返回结果
    return result
