import numpy as np
import matplotlib.pyplot as plt
import math as mt

def AdvDiff_solve(beta, kappa, source, bc_left, bc_right):

    # domain parameters
    x_A = 0.0
    L   = 1.0
    x_B = x_A + L

    nel = 100

    # mesh parameters
    nnp = nel+1
    dx  = L/nel
    # mesh points
    xmesh = np.empty(shape=(nnp,1), dtype=float)
    for inp in range(nnp):
        xmesh[inp] = x_A + inp*dx
    # mesh connectivity
    ien = np.empty(shape=(nel,2), dtype=int)
    for iel in range(nel):
        ien[iel][0] = iel
        ien[iel][1] = iel+1

    # Gauss quadrature rules
    nq  = 2
    xiq = np.array([-1.0/mt.sqrt(3), 1.0/mt.sqrt(3)])
    wq  = np.array([1.0, 1.0])

    # Local (Lienar FEM) Shape Function
    N  = np.empty(shape=(2,nq), dtype=float)
    dN = np.empty(shape=(2,nq), dtype=float)

    for iq in range(nq):
        N[0][iq]  = 0.5*(1.0-xiq[iq])
        N[1][iq]  = 0.5*(1.0+xiq[iq])
        dN[0][iq] = -0.5
        dN[1][iq] = 0.5
    #print(N[0][:])
    # print(dN)
    # print(dN[0,:].shape)
    # print(dN[:,0].shape)

    # declare global LHS matrix and RHS array
    A = np.zeros(shape=(nnp,nnp), dtype=float)
    b = np.zeros(shape=(nnp,1), dtype=float)

    # Loop over elements to assemble LHS matrix and RHS array
    for iel in range(nel):
        nshl = 2
        # element level LHS matrix and RHS array
        Ae = np.zeros(shape=(nshl,nshl), dtype=float)
        be = np.zeros(shape=(nshl,1), dtype=float)
        # element level nodes
        xl  = np.array([ xmesh[ien[iel][0]], xmesh[ien[iel][1]] ])
        Nq  = np.empty(shape=(nshl,1), dtype=float)
        dNq = np.empty(shape=(nshl,1), dtype=float)
        # Loop over quadrature points for integral calculations
        for iq in range(nq):
            # Local shape functions and derivatives at qps
            Nq[:,0]  = N[:,iq]
            dNq[:,0] = dN[:,iq]
            # Jacobian matrix at qp
            dxdxi = 0.0
            #xglobal = 0.0
            for ishl in range(nshl):
                dxdxi = dxdxi+dNq[ishl]*xl[ishl]
                #xglobal = xglobal + Nq[ishl]*xl[ishl]
            #print(dxdxi)
            # determinant of jacobian and same weighted with gauss qp wts
            detJ  = dxdxi
            WdetJ = detJ*wq[iq]
            # inverse of jacobian
            dxidx = 1.0/dxdxi
            dNdxq = np.empty(shape=(nshl,1), dtype=float)
            for ishl in range(nshl):
                dNdxq[ishl] = dNq[ishl]*dxidx
            # problem parameters at qps
            betaq   = beta
            kappaq  = kappa
            sourceq = source
            # VMS stabilization parameter at qps
            hhalf_mesh = detJ
            tauq = 1.0/mt.sqrt((betaq/hhalf_mesh)**2 + 9.0*(kappaq/hhalf_mesh**2)**2)
            #print(tauq)
            # Calculating element-wise contributions LHS matrix and RHS array
            for ishl in range(nshl):
                # local RHS array
                be[ishl] = be[ishl] + (Nq[ishl] + dNdxq[ishl]*betaq*tauq)*sourceq*WdetJ
                for jshl in range(nshl):
                    Ae[ishl][jshl] = Ae[ishl][jshl] + (Nq[ishl]*betaq*dNdxq[jshl] + dNdxq[ishl]*kappaq*dNdxq[jshl] + (dNdxq[ishl]*betaq)*tauq*(betaq*dNdxq[jshl]))*WdetJ

        # end loop over qps

        # Assemeble Ae and be to A and b respectively
        for ishl in range(nshl):
            b[ien[iel][ishl]] = b[ien[iel][ishl]] + be[ishl]
            for jshl in range(nshl):
                A[ien[iel][ishl]][ien[iel][jshl]] = A[ien[iel][ishl]][ien[iel][jshl]] + Ae[ishl][jshl]

    # end loop over elements

    # Account the BCs in the global assembled LHS matix and RHS vector
    b[0]     = bc_left
    b[nnp-1] = bc_right
    b[1]     = b[1] - A[1][0]*b[0]
    b[nnp-2] = b[nnp-2] - A[nnp-2][nnp-1]*b[nnp-1]

    A[0][0] = 1.0
    A[0][1] = 0.0
    A[1][0] = 0.0
    A[nnp-2][nnp-1] = 0.0
    A[nnp-1][nnp-2] = 0.0
    A[nnp-1][nnp-1] = 1.0

    u = np.linalg.solve(A,b)
    # fig = plt.figure()
    # plt.plot(xmesh,u,'r--')
    # plt.show()
    slope = (u[-2]-u[-1])/(xmesh[-2]-xmesh[-1])
    return -kappa*slope
