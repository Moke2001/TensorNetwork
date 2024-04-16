##  求解DMRG过程中的等效哈密顿量
import numpy as np
from Basis.Operator.Operator import Operator
from Basis.Operator.OperatorList import OperatorList
from Basis.State.MatrixProductState import MatrixProductState


def effective_hamiltonian(H_list,position,psi_origin):
    ##  参量类型检查模块-------------------------------------------------------------------------------------------------------------------

    assert isinstance(H_list,OperatorList) or isinstance(H_list,Operator), "H_list必须是OperatorList类型或Operator类型"
    assert isinstance(position,int),"position必须是int类型"
    assert isinstance(psi_origin,MatrixProductState),"psi_origin必须是MatrixProductState类型"

    ##  初始化
    psi=psi_origin.copy()  # 防止参量被改变
    result=0  # 结果初始化

    ##  单一算符计算模块-------------------------------------------------------------------------------------------------------------------

    if isinstance(H_list,Operator):

        ##  单个位点作用的情况----------------------------------------------------

        if isinstance(H_list.target_index,int):
            index_op = H_list.target_index  # 算符作用位点
            index_c=position  # 中心位点
            N=psi_origin.N

            ##  当作用在中心右边时----------------------------------------------------

            if index_op>index_c:

                ##  作用在右端，中心在左端
                if index_op==N-1 and index_c==0:
                    list_end=np.einsum('ab,db,ed->ae',psi[index_op],H_list.data,psi[index_op].conjugate())
                    dim_physics=psi[index_c].shape[0]  # 物理指标
                    dim_virtual=psi[index_c].shape[1]  # 虚拟指标
                    I_center=np.identity(dim_physics)
                    I_right=np.identity(dim_virtual)
                    list_start=np.einsum('ac,be,df->abcdef',I_center,I_right,I_right)
                    for i in range(index_c+1,index_op):
                        list_start=np.einsum('abcdef,egh,fij->abcdhj',list_start,psi[i],psi[i].conjugate())
                    result=np.einsum('abcdhj,hj->abcd',list_start,list_end)

                ##  作用在右端，中心不在左端
                elif index_op==N-1 and index_c!=0:
                    list_end = np.einsum('ab,db,ed->ae', psi[index_op], H_list.data,psi[index_op].conjugate())
                    dim_physics = psi[index_c].shape[1]  # 物理指标
                    dim_virtual_0 = psi[index_c].shape[0]  # 左虚拟指标
                    dim_virtual_1 = psi[index_c].shape[2]  # 右虚拟指标
                    I_left=np.identity(dim_virtual_0)
                    I_right=np.identity(dim_virtual_1)
                    I_center=np.identity(dim_physics)
                    list_start = np.einsum('ad,eb,cg,fh->abcdefgh',I_left,I_center,I_right,I_right)
                    for i in range(index_c + 1, index_op):
                        list_start = np.einsum('abcdefgh,gij,hkl->abcdefjl', list_start, psi[i], psi[i].conjugate())
                    result = np.einsum('abcdefjl,jl->abcdef', list_start, list_end)

                ##  作用不在右端，中心在左端
                elif index_op!=N-1 and index_c==0:
                    list_end = np.einsum('abc,db,edc->ae', psi[index_op], H_list.data,psi[index_op].conjugate())
                    dim_physics = psi[index_c].shape[0]  # 物理指标
                    dim_virtual = psi[index_c].shape[1]  # 虚拟指标
                    I_center = np.identity(dim_physics)
                    I_right = np.identity(dim_virtual)
                    list_start = np.einsum('ac,be,df->abcdef', I_center, I_right, I_right)
                    for i in range(index_c + 1, index_op):
                        list_start = np.einsum('abcdef,egh,fij->abcdhj', list_start, psi[i], psi[i].conjugate())
                    result = np.einsum('abcdhj,hj->abcd', list_start, list_end)

                ##  作用不在右端，中心不在左端
                else:
                    list_end = np.einsum('abc,db,edc->ae', psi[index_op], H_list.data,psi[index_op].conjugate())
                    dim_physics = psi[index_c].shape[1]  # 物理指标
                    dim_virtual_0 = psi[index_c].shape[0]  # 左虚拟指标
                    dim_virtual_1 = psi[index_c].shape[2]  # 右虚拟指标
                    I_left = np.identity(dim_virtual_0)
                    I_right = np.identity(dim_virtual_1)
                    I_center = np.identity(dim_physics)
                    list_start = np.einsum('ad,eb,cg,fh->abcdefgh', I_left, I_center, I_right, I_right)
                    for i in range(index_c + 1, index_op):
                        list_start = np.einsum('abcdefgh,gij,hkl->abcdefjl', list_start, psi[i], psi[i].conjugate())
                    result = np.einsum('abcdefjl,jl->abcdef', list_start, list_end)

            ##  当作用在中心左边时----------------------------------------------------

            elif index_op<index_c:

                ##  作用在左端，中心在右端
                if index_c==N-1 and index_op==0:
                    list_start=np.einsum('ab,ca,cd->bd',psi[index_op],H_list.data,psi[index_op].conjugate())
                    dim_physics=psi[index_c].shape[1]
                    dim_virtual = psi[index_c].shape[0]
                    I_left=np.identity(dim_virtual)
                    I_center=np.identity(dim_physics)
                    list_end=np.einsum('ab,cd,ef->afbced',I_left,I_center,I_left)
                    for i in range(index_op+1,index_c):
                        list_start=np.einsum('ab,acd,bce->de',list_start,psi[i],psi[i].conjugate())
                    result=np.einsum('af,afbced->bced',list_start,list_end)

                ##  作用不在左端，中心在右端
                elif index_c==N-1 and index_op!=0:
                    list_start = np.einsum('abc,eb,aef->cf', psi[index_op], H_list.data,psi[index_op].conjugate())
                    for i in range(index_op + 1, index_c):
                        list_start = np.einsum('ab,acd,bce->de', list_start, psi[i], psi[i].conjugate())
                    dim_physics = psi[index_c].shape[1]
                    dim_virtual = psi[index_c].shape[0]
                    I_left = np.identity(dim_virtual)
                    I_center = np.identity(dim_physics)
                    list_end = np.einsum('ab,cd,ef->afbced', I_left, I_center, I_left)
                    result=np.einsum('af,afbced->bced',list_start,list_end)

                ##  作用在左端，中心不在右端
                elif index_c!=N-1 and index_op==0:
                    list_start = np.einsum('ab,ca,cd->bd', psi[index_op], H_list.data,psi[index_op].conjugate())
                    dim_physics = psi[index_c].shape[1]
                    dim_virtual_0 = psi[index_c].shape[0]
                    dim_virtual_1 = psi[index_c].shape[2]
                    I_left = np.identity(dim_virtual_0)
                    I_center = np.identity(dim_physics)
                    I_right = np.identity(dim_virtual_1)
                    list_end= np.einsum('ab,cd,gh,ef->afbhcegd', I_left,I_right,I_center,I_left)
                    for i in range(index_op + 1, index_c):
                        list_start = np.einsum('ab,acd,bce->de', list_start, psi[i], psi[i].conjugate())
                    result = np.einsum('af,afbhcegd->bhcegd', list_start, list_end)

                ##  作用不在左端，中心不在右端
                else:
                    list_start = np.einsum('abc,eb,aef->cf', psi[index_op], H_list.data,psi[index_op].conjugate())
                    for i in range(index_op + 1, index_c):
                        list_start = np.einsum('ab,acd,bce->de', list_start, psi[i], psi[i].conjugate())
                    dim_physics = psi[index_c].shape[1]
                    dim_virtual_0 = psi[index_c].shape[0]
                    dim_virtual_1 = psi[index_c].shape[2]
                    I_left = np.identity(dim_virtual_0)
                    I_center = np.identity(dim_physics)
                    I_right = np.identity(dim_virtual_1)
                    list_end = np.einsum('ab,cd,gh,ef->afbhcegd', I_left, I_right, I_center, I_left)
                    result = np.einsum('af,afbhcegd->bhcegd', list_start, list_end)

            ##  作用与中心重合时----------------------------------------------------

            else:

                ##  在左端
                if index_c==0:
                    dim_virtual=psi[index_c].shape[1]
                    I_right=np.identity(dim_virtual)
                    result=np.einsum('be,fc->efbc',H_list.data, I_right)

                ##  在右端
                elif  index_c==N-1:
                    dim_virtual = psi[index_c].shape[0]
                    I_left = np.identity(dim_virtual)
                    result = np.einsum('ad,be->deab', I_left, H_list.data)

                ##  在中间
                else:
                    dim_virtual_0 = psi[index_c].shape[0]
                    dim_virtual_1 = psi[index_c].shape[2]
                    I_left = np.identity(dim_virtual_0)
                    I_right = np.identity(dim_virtual_1)
                    result = np.einsum('ad,be,cf->defabc', I_left, H_list.data, I_right)

        ##  两个位点作用情况----------------------------------------------------

        elif isinstance(H_list.target_index,list):
            index_0=H_list.target_index[0]
            index_1=H_list.target_index[1]
            index_n=position
            N= psi_origin.N

            ##  两者重合时----------------------------------------------------

            if index_0 == index_n or index_1 == index_n:

                ##  在两端
                if index_1 == N-1 and index_0==0:
                    if position==0:
                        s='bc,deac,fe->abdf'
                        result=np.einsum(s,psi[index_1],H_list.data,psi[index_1].conjugate())
                    elif position==N-1:
                        s='ab,deac,df->bcfe'
                        result = np.einsum(s, psi[index_0], H_list.data, psi[index_0].conjugate())

                ##  在左端
                elif index_1 != N-1 and index_0==0:
                    if position == 0:
                        s = 'bcg,deac,feg->abdf'
                        result = np.einsum(s, psi[index_1], H_list.data, psi[index_1].conjugate())
                    else:
                        s = 'ab,deac,df,gh->bcgfeh'
                        result = np.einsum(s, psi[index_0], H_list.data, psi[index_0].conjugate(),np.identity(psi[index_1].shape[2]))

                ##  在右端
                elif index_1 == N-1 and index_0!=0:
                    if position == N-1:
                        s='gab,deac,gdf->bcfe'
                        result = np.einsum(s, psi[index_0], H_list.data, psi[index_0].conjugate())
                    else:
                        s = 'bc,deac,fe,gh->gabhdf'
                        result = np.einsum(s, psi[index_1], H_list.data, psi[index_1].conjugate(), np.identity(psi[index_0].shape[0]))

                ##  在中间
                else:
                    if position == index_0:
                        s='hcd,febc,ied,ga->abhgfi'
                        result=np.einsum(s,psi[index_1],H_list.data, psi[index_1].conjugate(),np.identity(psi[index_0].shape[0]))
                    else:
                        s='abc,hibd,ahg,fe->cdegif'
                        result = np.einsum(s, psi[index_0], H_list.data, psi[index_0].conjugate(), np.identity(psi[index_1].shape[2]))

            else:

                ##  作用在中心右端----------------------------------------------------

                if index_0>index_n:

                    ##  作用在右端，中心在左端
                    if index_1==N-1 and index_n==0:
                        dim_physics = psi[index_n].shape[0]  # 物理指标
                        dim_virtual = psi[index_n].shape[1]  # 虚拟指标
                        I_center = np.identity(dim_physics)
                        I_right = np.identity(dim_virtual)
                        list_start = np.einsum('ac,be,df->abcdef', I_center, I_right, I_right)
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('abcdef,egh,fij->abcdhj', list_start, psi[i], psi[i].conjugate())
                        s='abc,cd,gebd,hgf,fe->ah'
                        list_end=np.einsum(s,psi[index_0],psi[index_1],H_list.data,psi[index_0].conjugate(),psi[index_1].conjugate())
                        result = np.einsum('abcdhj,hj->abcd', list_start, list_end)

                    ##  作用不在右端，中心在左端
                    elif index_1!=N-1 and index_n==0:
                        dim_physics = psi[index_n].shape[0]  # 物理指标
                        dim_virtual = psi[index_n].shape[1]  # 虚拟指标
                        I_center = np.identity(dim_physics)
                        I_right = np.identity(dim_virtual)
                        list_start = np.einsum('ac,be,df->abcdef', I_center, I_right, I_right)
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('abcdef,egh,fij->abcdhj', list_start, psi[i], psi[i].conjugate())
                        s = 'abc,cdi,gebd,hgf,fei->ah'
                        list_end = np.einsum(s, psi[index_0], psi[index_1], H_list.data,psi[index_0].conjugate(), psi[index_1].conjugate())
                        result = np.einsum('abcdhj,hj->abcd', list_start, list_end)

                    ##  作用在右端，中心不在左端
                    elif index_1 == N - 1 and index_n != 0:
                        s = 'abc,cd,gebd,hgf,fe->ah'
                        list_end = np.einsum(s, psi[index_0], psi[index_1], H_list.data,psi[index_0].conjugate(), psi[index_1].conjugate())
                        dim_physics = psi[index_n].shape[1]  # 物理指标
                        dim_virtual_0 = psi[index_n].shape[0]  # 左虚拟指标
                        dim_virtual_1 = psi[index_n].shape[2]  # 右虚拟指标
                        I_left = np.identity(dim_virtual_0)
                        I_right = np.identity(dim_virtual_1)
                        I_center = np.identity(dim_physics)
                        list_start = np.einsum('ad,eb,cg,fh->abcdefgh', I_left, I_center, I_right, I_right)
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('abcdefgh,gij,hkl->abcdefjl', list_start, psi[i], psi[i].conjugate())
                        result = np.einsum('abcdefjl,jl->abcdef', list_start, list_end)

                    ##  作用不在右端，中心不在左端
                    else:
                        s = 'abc,cdi,gebd,hgf,fei->ah'
                        list_end = np.einsum(s, psi[index_0], psi[index_1], H_list.data,psi[index_0].conjugate(), psi[index_1].conjugate())
                        dim_physics = psi[index_n].shape[1]  # 物理指标
                        dim_virtual_0 = psi[index_n].shape[0]  # 左虚拟指标
                        dim_virtual_1 = psi[index_n].shape[2]  # 右虚拟指标
                        I_left = np.identity(dim_virtual_0)
                        I_right = np.identity(dim_virtual_1)
                        I_center = np.identity(dim_physics)
                        list_start = np.einsum('ad,eb,cg,fh->abcdefgh', I_left, I_center, I_right, I_right)
                        for i in range(index_n + 1, index_0):
                            list_start = np.einsum('abcdefgh,gij,hkl->abcdefjl', list_start, psi[i], psi[i].conjugate())
                        result = np.einsum('abcdefjl,jl->abcdef', list_start, list_end)

                ##  作用在中心左端时----------------------------------------------------

                else:

                    ##  作用在左端，中心在右端
                    if index_n==N-1 and index_0==0:
                        s='ab,bcd,efac,eg,gfh->dh'
                        list_start=np.einsum(s,psi[index_0],psi[index_1],H_list.data,psi[index_0].conjugate(),psi[index_1].conjugate())
                        for i in range(index_1+1,index_n):
                            list_start = np.einsum('ab,acd,bce->de', list_start, psi[i], psi[i].conjugate())
                        dim_physics = psi[index_n].shape[0]
                        dim_virtual = psi[index_n].shape[1]
                        I_left = np.identity(dim_virtual)
                        I_center = np.identity(dim_physics)
                        list_end = np.einsum('ab,cd,ef->afbced', I_left, I_center, I_left)
                        result = np.einsum('af,afbced->bced', list_start, list_end)

                    ##  作用在左端，中心不在右端
                    elif index_n!=N-1 and index_0==0:
                        s = 'ab,bcd,egac,ef,fgh->dh'
                        list_start = np.einsum(s, psi[index_0], psi[index_1], H_list.data,psi[index_0].conjugate(), psi[index_1].conjugate())
                        for i in range(index_1+1,index_n):
                            list_start = np.einsum('ab,acd,bce->de', list_start, psi[i], psi[i].conjugate())
                        dim_physics = psi[index_n].shape[1]
                        dim_virtual_0 = psi[index_n].shape[0]
                        dim_virtual_1 = psi[index_n].shape[2]
                        I_left = np.identity(dim_virtual_0)
                        I_center = np.identity(dim_physics)
                        I_right = np.identity(dim_virtual_1)
                        list_end = np.einsum('ab,cd,gh,ef->afbhcegd', I_left, I_right, I_center, I_left)
                        result = np.einsum('af,afbhcegd->bhcegd', list_start, list_end)

                    ##  作用不在左端，中心在右端
                    elif index_n == N - 1 and index_0 != 0:
                        s = 'iab,bcd,efac,ieg,gfh->dh'
                        list_start = np.einsum(s, psi[index_0], psi[index_1], H_list.data,psi[index_0].conjugate(), psi[index_1].conjugate())
                        for i in range(index_1+1,index_n):
                            list_start = np.einsum('ab,acd,bce->de', list_start, psi[i], psi[i].conjugate())
                        dim_physics = psi[index_n].shape[1]
                        dim_virtual = psi[index_n].shape[0]
                        I_left = np.identity(dim_virtual)
                        I_center = np.identity(dim_physics)
                        list_end = np.einsum('ab,cd,ef->afbced', I_left, I_center, I_left)
                        result = np.einsum('af,afbced->bced', list_start, list_end)

                    ##  作用不在左端，中心不在右端
                    else:
                        s = 'iab,bcd,efac,ieg,gfh->dh'
                        list_start = np.einsum(s, psi[index_0], psi[index_1], H_list.data,psi[index_0].conjugate(), psi[index_1].conjugate())
                        for i in range(index_1+1,index_n):
                            list_start = np.einsum('ab,acd,bce->de', list_start, psi[i], psi[i].conjugate())
                        dim_physics = psi[index_n].shape[1]
                        dim_virtual_0 = psi[index_n].shape[0]
                        dim_virtual_1 = psi[index_n].shape[2]
                        I_left = np.identity(dim_virtual_0)
                        I_center = np.identity(dim_physics)
                        I_right = np.identity(dim_virtual_1)
                        list_end = np.einsum('ab,cd,gh,ef->afbhcegd', I_left, I_right, I_center, I_left)
                        result = np.einsum('af,afbhcegd->bhcegd', list_start, list_end)

    ##  算符列表递归模块-------------------------------------------------------------------------------------------------------------------

    elif isinstance(H_list,OperatorList):

        ##  单位点算符期望值的循环
        for i in range(len(H_list.single_list)):
            result=result+effective_hamiltonian(H_list.single_list[i],position,psi)

        ##  双位点算符期望值的循环
        for i in range(len(H_list.double_list)):
            result = result + effective_hamiltonian(H_list.double_list[i],position,psi)

    ##  结果返回模块-----------------------------------------------------------------------------------------------------------------------

    ##  返回结果
    return result