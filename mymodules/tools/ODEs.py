# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 14:44:48 2023

@author: fkogel

v3.1.0

This Module contains the definitions of all ordinary differential equations (ODEs)
which are used in scipy's solve_ivp function e.g. within the rate and optical
Bloch equations (e.g. :meth:`~System.System.calc_OBEs`).
"""
import numpy as np
from scipy.constants import c,h,hbar,pi,g,physical_constants
from scipy.constants import k as k_B
from scipy.constants import u as u_mass
from math import floor
from numba import jit, prange
#%%
@jit(nopython=True,parallel=False,fastmath=True) #original (slow) ODE form from the Fokker-Planck paper
def ode0_OBEs(T,y_vec,lNum,uNum,pNum,G,f,om_eg,om_k,betaB,dMat,muMat,M_indices,h_gek,h_gege,phi):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex64)
    
    ymat    = np.zeros((N,N),dtype=np.complex64)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(lNum):
        for b in range(uNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c in range(lNum):
                        dymat[a,lNum+b] += 1j*G[k]*f[k,q+1]/2/(2**0.5) *h_gek[c,b,k]*np.exp(1j*phi[k]+1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[a,c]
                    for c_ in range(uNum):
                        dymat[a,lNum+b] -= 1j*G[k]*f[k,q+1]/2/(2**0.5) *h_gek[a,c_,k]*np.exp(1j*phi[k]+1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,lNum+b]
                for n in M_indices[1][b]:
                    dymat[a,lNum+b] += 1j*(-1.)**q* betaB[q+1]* muMat[1][b,n,-q+1]* ymat[a,lNum+n]
                for m in M_indices[0][a]:
                    dymat[a,lNum+b] -= 1j*(-1.)**q* betaB[q+1]* muMat[0][m,a,-q+1]* ymat[m,lNum+b]
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]
    for a in range(uNum):
        for b in range(a,uNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c in range(lNum):
                        dymat[lNum+a,lNum+b] += 1j*G[k]/2/(2**0.5)*(
                            f[k,q+1] *h_gek[c,b,k]*np.exp(1j*phi[k]+1j*T*(om_eg[c,b]-om_k[k]))* dMat[c,b,q+1]* ymat[lNum+a,c]
                            - np.conj(f[k,q+1]) *h_gek[c,a,k]*np.exp(-1j*phi[k]-1j*T*(om_eg[c,a]-om_k[k]))* dMat[c,a,q+1]* ymat[c,lNum+b])
                for n in M_indices[1][b]:
                    dymat[lNum+a,lNum+b] += 1j*(-1.)**q* betaB[q+1]* muMat[1][b,n,-q+1]* ymat[lNum+a,lNum+n]
                for m in M_indices[1][a]:
                    dymat[lNum+a,lNum+b] -= 1j*(-1.)**q* betaB[q+1]* muMat[1][m,a,-q+1]* ymat[lNum+m,lNum+b]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for a in range(lNum):
        for b in range(a,lNum):
            for q in [-1,0,1]:
                for k in range(pNum):
                    for c_ in range(uNum):
                        dymat[a,b] -= 1j*G[k]/2/(2**0.5)*(
                            f[k,q+1] *h_gek[a,c_,k]*np.exp(1j*phi[k]+1j*T*(om_eg[a,c_]-om_k[k]))* dMat[a,c_,q+1]* ymat[lNum+c_,b]
                            - np.conj(f[k,q+1]) *h_gek[b,c_,k]*np.exp(-1j*phi[k]-1j*T*(om_eg[b,c_]-om_k[k]))* dMat[b,c_,q+1]* ymat[a,lNum+c_])
                for n in M_indices[0][b]:
                    dymat[a,b] += 1j*(-1.)**q* betaB[q+1]* muMat[0][b,n,-q+1]* ymat[a,n]
                for m in M_indices[0][a]:
                    dymat[a,b] -= 1j*(-1.)**q* betaB[q+1]* muMat[0][m,a,-q+1]* ymat[m,b]
                for c_ in range(uNum):
                    for c__ in range(uNum):
                        dymat[a,b] += dMat[a,c_,q+1]* dMat[b,c__,q+1] *h_gege[a,c_,b,c__]*np.exp(1j*T*(om_eg[a,c_]-om_eg[b,c__]))* ymat[lNum+c_,lNum+c__]
    
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec
#%%
@jit(nopython=True,parallel=False,fastmath=True) #same as ode1_OBEs_opt2 but compatible with two different electr. ex. states
def ode1_OBEs_opt3(T,y_vec,lNum,uNum,pNum,M_indices,Gfd,om_gek,betamu,dd,ck_indices,Gam_fac):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex128)
    
    ymat    = np.zeros((N,N),dtype=np.complex128)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(uNum):
        for c,k in zip(ck_indices[1][a][0],ck_indices[1][a][1]):
            for b in range(lNum):
                dymat[b,lNum+a] += Gfd[c,a,k]* np.exp(1j*om_gek[c,a,k]*T)* ymat[b,c]
            for b in range(a,uNum):            
                dymat[lNum+a,lNum+b] += np.conj(Gfd[c,a,k])* np.exp(-1j*om_gek[c,a,k]*T)* ymat[c,lNum+b]
        for n in M_indices[1][a]:
            for b in range(lNum):
                dymat[b,lNum+a] += betamu[1][a,n] * ymat[b,lNum+n]
        for m in M_indices[1][a]:
            for b in range(a,uNum):
                dymat[lNum+a,lNum+b] -= betamu[1][m,a] * ymat[lNum+m,lNum+b]
        for b in range(a,uNum):
            for n in M_indices[1][b]:
                dymat[lNum+a,lNum+b] += betamu[1][b,n] * ymat[lNum+a,lNum+n]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]*(Gam_fac[a]+Gam_fac[b])/2  #!!!!!!!!!!!!
    for b in range(uNum-1,-1,-1):       
        for c,k in zip(ck_indices[1][b][0],ck_indices[1][b][1]):
            for a in range(0,b+1):
                dymat[lNum+a,lNum+b] += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)* ymat[lNum+a,c]

    for a in range(lNum):
        for c_,k in zip(ck_indices[0][a][0],ck_indices[0][a][1]):
            for b in range(uNum):
                dymat[a,lNum+b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,lNum+b]
            for b in range(a,lNum):    
                dymat[a,b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,b]
        for m in M_indices[0][a]:
            for b in range(uNum):
                dymat[a,lNum+b] -= betamu[0][m,a] * ymat[m,lNum+b]
            for b in range(a,lNum):    
                dymat[a,b] -= betamu[0][m,a] * ymat[m,b]
        for b in range(uNum):
            dymat[a,lNum+b] -= 0.5*Gam_fac[b]*ymat[a,lNum+b]                     #!!!!!!!!!!!!   
        for b in range(a,lNum):
            for n in M_indices[0][b]:
                dymat[a,b] += betamu[0][b,n] * ymat[a,n]
            for c_ in range(uNum):
                for c__ in range(uNum):
                    dymat[a,b] += dd[a,c_,b,c__] * np.exp(1j*T*(om_gek[a,c_,0]-om_gek[b,c__,0])) * ymat[lNum+c_,lNum+c__] *(Gam_fac[c_]+Gam_fac[c__])/2#!!!!!!!!!!!!????
    for b in range(lNum-1,-1,-1):
        for c_,k in zip(ck_indices[0][b][0],ck_indices[0][b][1]):
            for a in range(0,b+1):
                dymat[a,b] -= np.conj(Gfd[b,c_,k])* np.exp(-1j*om_gek[b,c_,k]*T)* ymat[a,lNum+c_]
                
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=True) #same as ode1_OBEs_opt1 but further optimized by rearranging the loops
def ode1_OBEs_opt2(T,y_vec,lNum,uNum,pNum,M_indices,Gfd,om_gek,betamu,dd,ck_indices):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex128)
    
    ymat    = np.zeros((N,N),dtype=np.complex128)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(uNum):
        for c,k in zip(ck_indices[1][a][0],ck_indices[1][a][1]):
            for b in range(lNum):
                dymat[b,lNum+a] += Gfd[c,a,k]* np.exp(1j*om_gek[c,a,k]*T)* ymat[b,c]
            for b in range(a,uNum):            
                dymat[lNum+a,lNum+b] += np.conj(Gfd[c,a,k])* np.exp(-1j*om_gek[c,a,k]*T)* ymat[c,lNum+b]
        for n in M_indices[1][a]:
            for b in range(lNum):
                dymat[b,lNum+a] += betamu[1][a,n] * ymat[b,lNum+n]
        for m in M_indices[1][a]:
            for b in range(a,uNum):
                dymat[lNum+a,lNum+b] -= betamu[1][m,a] * ymat[lNum+m,lNum+b]
        for b in range(a,uNum):
            for n in M_indices[1][b]:
                dymat[lNum+a,lNum+b] += betamu[1][b,n] * ymat[lNum+a,lNum+n]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]                     #!!!!!!!!!!!!
    for b in range(uNum-1,-1,-1):       
        for c,k in zip(ck_indices[1][b][0],ck_indices[1][b][1]):
            for a in range(0,b+1):
                dymat[lNum+a,lNum+b] += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)* ymat[lNum+a,c]

    for a in range(lNum):
        for c_,k in zip(ck_indices[0][a][0],ck_indices[0][a][1]):
            for b in range(uNum):
                dymat[a,lNum+b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,lNum+b]
            for b in range(a,lNum):    
                dymat[a,b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,b]
        for m in M_indices[0][a]:
            for b in range(uNum):
                dymat[a,lNum+b] -= betamu[0][m,a] * ymat[m,lNum+b]
            for b in range(a,lNum):    
                dymat[a,b] -= betamu[0][m,a] * ymat[m,b]
        for b in range(uNum):
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]                        #!!!!!!!!!!!!   
        for b in range(a,lNum):
            for n in M_indices[0][b]:
                dymat[a,b] += betamu[0][b,n] * ymat[a,n]
            for c_ in range(uNum):
                for c__ in range(uNum):
                    dymat[a,b] += dd[a,c_,b,c__] * np.exp(1j*T*(om_gek[a,c_,0]-om_gek[b,c__,0])) * ymat[lNum+c_,lNum+c__] #!!!!!!!!!!!!
    for b in range(lNum-1,-1,-1):
        for c_,k in zip(ck_indices[0][b][0],ck_indices[0][b][1]):
            for a in range(0,b+1):
                dymat[a,b] -= np.conj(Gfd[b,c_,k])* np.exp(-1j*om_gek[b,c_,k]*T)* ymat[a,lNum+c_]
                
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=True) #same as ode1_OBEs but in optimized form with ck_indices variable
def ode1_OBEs_opt1(T,y_vec,lNum,uNum,pNum,M_indices,Gfd,om_gek,betamu,dd,ck_indices):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex128)
    
    ymat    = np.zeros((N,N),dtype=np.complex128)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(lNum):
        for b in range(uNum):
            for c,k in zip(*ck_indices[1][b]):
                dymat[a,lNum+b] += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)* ymat[a,c]
            for c_,k in zip(*ck_indices[0][a]):
                dymat[a,lNum+b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,lNum+b]
            for n in M_indices[1][b]:
                dymat[a,lNum+b] += betamu[1][b,n] * ymat[a,lNum+n]
            for m in M_indices[0][a]:
                dymat[a,lNum+b] -= betamu[0][m,a] * ymat[m,lNum+b]
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]
    for a in range(uNum):
        for b in range(a,uNum):
            for c,k in zip(*ck_indices[1][b]):
                dymat[lNum+a,lNum+b] += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)* ymat[lNum+a,c]
            for c,k in zip(*ck_indices[1][a]):
                dymat[lNum+a,lNum+b] += np.conj(Gfd[c,a,k])* np.exp(-1j*om_gek[c,a,k]*T)* ymat[c,lNum+b]
            for n in M_indices[1][b]:
                dymat[lNum+a,lNum+b] += betamu[1][b,n] * ymat[lNum+a,lNum+n]
            for m in M_indices[1][a]:
                dymat[lNum+a,lNum+b] -= betamu[1][m,a] * ymat[lNum+m,lNum+b]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for a in range(lNum):
        for b in range(a,lNum):
            for c_,k in zip(*ck_indices[0][a]):
                dymat[a,b] -= Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)* ymat[lNum+c_,b]
            for c_,k in zip(*ck_indices[0][b]):
                dymat[a,b] -= np.conj(Gfd[b,c_,k])* np.exp(-1j*om_gek[b,c_,k]*T)* ymat[a,lNum+c_]
            for n in M_indices[0][b]:
                dymat[a,b] += betamu[0][b,n] * ymat[a,n]
            for m in M_indices[0][a]:
                dymat[a,b] -= betamu[0][m,a] * ymat[m,b]
            for c_ in range(uNum):
                for c__ in range(uNum):
                    dymat[a,b] += dd[a,c_,b,c__] * np.exp(1j*T*(om_gek[a,c_,0]-om_gek[b,c__,0])) * ymat[lNum+c_,lNum+c__]
    
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec

#%%
@jit(nopython=True,parallel=False,fastmath=True) #same as ode0_OBEs but in optimized form with less input variables
def ode1_OBEs(T,y_vec,lNum,uNum,pNum,M_indices,Gfd,om_gek,betamu,dd):
    N       = lNum+uNum
    dymat   = np.zeros((N,N),dtype=np.complex128)
    
    ymat    = np.zeros((N,N),dtype=np.complex128)
    count   = 0
    for i in range(N):
        for j in range(i,N):
            ymat[i,j] = y_vec[count] + 1j* y_vec[count+1]
            count += 2     
    ymat    += np.conj(ymat.T) #is diagonal remaining purely real or complex?
    for index in range(N):
        ymat[index,index] *=0.5
    
    for a in range(lNum):
        for b in range(uNum):
            for c in range(lNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)
                dymat[a,lNum+b] += tmp* ymat[a,c]
            for c_ in range(uNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)
                dymat[a,lNum+b] -= tmp* ymat[lNum+c_,lNum+b]
            for n in M_indices[1][b]:
                dymat[a,lNum+b] += betamu[1][b,n] * ymat[a,lNum+n]
            for m in M_indices[0][a]:
                dymat[a,lNum+b] -= betamu[0][m,a] * ymat[m,lNum+b]
            dymat[a,lNum+b] -= 0.5*ymat[a,lNum+b]
    for a in range(uNum):
        for b in range(a,uNum):
            for c in range(lNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[c,b,k]* np.exp(1j*om_gek[c,b,k]*T)
                dymat[lNum+a,lNum+b] += tmp* ymat[lNum+a,c]
            for c in range(lNum):
                tmp = 0
                for k in range(pNum):
                    tmp += np.conj(Gfd[c,a,k])* np.exp(-1j*om_gek[c,a,k]*T)
                dymat[lNum+a,lNum+b] += tmp* ymat[c,lNum+b]
            for n in M_indices[1][b]:
                dymat[lNum+a,lNum+b] += betamu[1][b,n] * ymat[lNum+a,lNum+n]
            for m in M_indices[1][a]:
                dymat[lNum+a,lNum+b] -= betamu[1][m,a] * ymat[lNum+m,lNum+b]
            dymat[lNum+a,lNum+b] -= ymat[lNum+a,lNum+b]
    for a in range(lNum):
        for b in range(a,lNum):
            for c_ in range(uNum):
                tmp = 0
                for k in range(pNum):
                    tmp += Gfd[a,c_,k]* np.exp(1j*om_gek[a,c_,k]*T)
                dymat[a,b] -= tmp* ymat[lNum+c_,b]
            for c_ in range(uNum):
                tmp = 0
                for k in range(pNum):
                    tmp += np.conj(Gfd[b,c_,k])* np.exp(-1j*om_gek[b,c_,k]*T)
                dymat[a,b] -= tmp* ymat[a,lNum+c_]
            for n in M_indices[0][b]:
                dymat[a,b] += betamu[0][b,n] * ymat[a,n]
            for m in M_indices[0][a]:
                dymat[a,b] -= betamu[0][m,a] * ymat[m,b]
            for c_ in range(uNum):
                for c__ in range(uNum):
                    dymat[a,b] += dd[a,c_,b,c__] * np.exp(1j*T*(om_gek[a,c_,0]-om_gek[b,c__,0])) * ymat[lNum+c_,lNum+c__]
    
    dy_vec = np.zeros( N*(N+1) )
    count = 0
    for i in range(N):
        for j in range(i,N):
            dy_vec[count]   = dymat[i,j].real
            dy_vec[count+1] = dymat[i,j].imag
            count += 2

    return dy_vec
#%%
@jit(nopython=True,parallel=False,fastmath=False)
def ode0_rateeqs_jit(t,N,lNum,uNum,pNum,Gamma,r,R1sum,R2sum,tswitch,M):
    dNdt = np.zeros(lNum+uNum)
    if floor(t/tswitch)%2 == 1: R_sum=R1sum
    else: R_sum=R2sum
    
    for l in prange(lNum):
        for u in prange(uNum):
            dNdt[l] += Gamma[u]* r[l,u] * N[lNum+u] + R_sum[l,u] * (N[lNum+u] - N[l])
        if not M.size == 0:
            for k in prange(lNum):
                dNdt[l] -= M[l,k] * (N[l]-N[k])
    for u in prange(uNum):
        dNdt[lNum+u]  = -Gamma[u]*N[lNum+u]
        for l in prange(lNum):
            dNdt[lNum+u] += R_sum[l,u] * (N[l] - N[lNum+u])
                          
    return dNdt

#%%
def ode0_rateeqs(t,N,lNum,uNum,pNum,Gamma,r,R1sum,R2sum,tswitch,M):
    dNdt = np.zeros(lNum+uNum)
    if floor(t/tswitch)%2 == 1: R_sum=R1sum
    else: R_sum=R2sum
    
    Nlu_matr = np.subtract.outer(N[:lNum], N[lNum:lNum+uNum])  
    
    dNdt[:lNum] = - np.sum(R_sum*Nlu_matr, axis=1) + Gamma*np.dot(r,N[lNum:lNum+uNum])
    if not M.size == 0:
        dNdt[:lNum] -= np.sum(M * np.subtract.outer(N[:lNum], N[:lNum]), axis=1)
    
    dNdt[lNum:lNum+uNum] = -Gamma*N[lNum:lNum+uNum] + np.sum(R_sum*Nlu_matr, axis=0)
              
    return dNdt

#%%
def ode1_rateeqs(t,y,lNum,uNum,pNum,Gamma,r,rx1,rx2,delta,sp_,w,k,r_k,m,tswitch,M,pos_dep):    
    dydt = np.zeros(lNum+uNum+3+3)
    if floor(t/tswitch)%2 == 1: rx=rx1
    else: rx=rx2
    sp = sp_.copy()
    # position dependent Force on particle due to Gaussian shape of Laserbeam:
    if pos_dep:
        for p in range(pNum):
            #if abs(y[-3]) > 1e-3: sp[p] = 1e-1
            d = np.linalg.norm(np.cross( y[-3:]-r_k[p] , k[p]/np.linalg.norm(k[p]) ))
            # r2 = np.dot(y[-3:], y[-3:]) - (np.dot(k[p], y[-3:])/np.linalg.norm(k[p]))**2
            sp[p] = sp[p] * np.exp(-2 * d**2 / ((0.2*w[p])**2) )
    
    # shape of k: (pNum,3)
    # shape of rx = (lNum,uNum,pNum), sp.shape = (pNum) ==> (rx*sp).shape = (lNum,uNum,pNum)
    # R = Gamma/2 * (rx*sp) / ( 1+4*(delta)**2/Gamma**2 )
    R = Gamma/2 * (rx*sp) / ( 1+4*( delta - np.dot(k,y[lNum+uNum:lNum+uNum+3]) )**2/Gamma**2 )    
    # sum R over pNum
    R_sum = np.sum(R,axis=2)
    # shape(Nlu_matr) = (lNum,uNum)
    Nlu_matr = y[:lNum,None] - y[None,lNum:lNum+uNum]#np.subtract.outer(y[:lNum], y[lNum:lNum+uNum])

    # __________ODE:__________
    # N_l' = ...
    dydt[:lNum] = - np.sum(R_sum*Nlu_matr, axis=1) + Gamma*np.dot(r,y[lNum:lNum+uNum])
    if not M.size == 0: # magnetic remixing of the ground states
        dydt[:lNum] -= np.sum(M * np.subtract.outer(y[:lNum], y[:lNum]), axis=1)
    # N_u' = ... 
    dydt[lNum:lNum+uNum] = -Gamma*y[lNum:lNum+uNum] + np.sum(R_sum*Nlu_matr, axis=0)
    # v' = ...
    dydt[lNum+uNum:lNum+uNum+3] = hbar/m * np.sum(np.dot(R,k) * Nlu_matr[:,:,None], axis=(0,1)) #+ g 
    # r' = ...    
    dydt[lNum+uNum+3:lNum+uNum+3+3] = y[lNum+uNum:lNum+uNum+3]
              
    return dydt

#%%
@jit(nopython=True,parallel=False,fastmath=False)
def ode1_rateeqs_jit(t,y,lNum,uNum,pNum,Gamma,r,rx1,rx2,delta,sp_,w,w_cyl,k,kabs,r_k,r_cyl_trunc,dir_cyl,m,tswitch,M,pos_dep,beta):    
    dydt = np.zeros(lNum+uNum+3+3)
    if floor(t/tswitch)%2 == 1: rx=rx1
    else: rx=rx2
    sp = sp_.copy()
    # position dependent Force on particle due to Gaussian shape of Laserbeam:
    if pos_dep:
        for p in range(pNum):
            r_ = y[-3:] - r_k[p]
            if w_cyl[p] != 0.0: # calculation for a beam which is widened by a cylindrical lens
                d2_w = np.dot(dir_cyl[p],r_)**2
                if d2_w > r_cyl_trunc[p]**2: #test if position is larger than the truncation radius along the dir_cyl direction
                    sp[:,:,p] = 0.0  
                else:
                    d2 = np.dot(np.cross(dir_cyl[p],k[p]/kabs[p]),r_)**2
                    sp[:,:,p] *= np.exp(-2*(d2_w/w_cyl[p]**2 + d2/w[p]**2))
            else: 
                r_perp = np.cross( r_ , k[p]/kabs[p] )
                sp[:,:,p] *= np.exp(-2 * np.dot(r_perp,r_perp) / w[p]**2 )  

    delta_ = delta + 2*pi*beta*t #frequency chirping
    # shape of k: (pNum,3)
    # shape of rx = (lNum,uNum,pNum), sp.shape = (pNum) ==> (rx*sp).shape = (lNum,uNum,pNum)
    # R = Gamma/2 * (rx*sp) / ( 1+4*(delta)**2/Gamma**2 )
    R = Gamma/2 * (rx*sp) / ( 1+4*( delta_ - np.dot(k,y[lNum+uNum:lNum+uNum+3]) )**2/(Gamma**2) ) 
    # sum R over pNum
    R_sum = np.sum(R,axis=2)
    
    # __________ODE:__________
    # N_l' = ...
    for l in range(lNum):
        for u in range(uNum):
            dydt[l] += Gamma[0,u,0]* r[l,u] * y[lNum+u] + R_sum[l,u] * (y[lNum+u] - y[l])
    if not M.size == 0:
        for l1 in range(lNum):
            for l2 in range(lNum):
                dydt[l1] -= M[l1,l2] * (y[l1]-y[l2])
    # N_u' = ... 
    for u in range(uNum):
        dydt[lNum+u]  = -Gamma[0,u,0]*y[lNum+u]
        for l in range(lNum):
            dydt[lNum+u] += R_sum[l,u] * (y[l] - y[lNum+u])
    # v' = ...
    for i in range(3):
        for l in range(lNum):
            for u in range(uNum):
                for p in range(pNum):
                    dydt[lNum+uNum+i] +=  hbar/m * k[p,i] * R[l,u,p] * ( y[l] - y[lNum+u] )
    # r' = ...    
    dydt[lNum+uNum+3:lNum+uNum+3+3] = y[lNum+uNum:lNum+uNum+3]
              
    return dydt

#%%
@jit(nopython=True,parallel=False,fastmath=False)
def ode1_rateeqs_jit_testI(t,y,lNum,uNum,pNum,Gamma,r,rx1,rx2,delta,sp_,k,m,tswitch,M,pos_dep,beta,I_tot):    
    dydt = np.zeros(lNum+uNum+3+3)
    if floor(t/tswitch)%2 == 1: rx=rx1
    else: rx=rx2
    sp = sp_.copy()
    # position dependent Force on particle due to Gaussian shape of Laserbeam:
    if pos_dep:
        sp *= I_tot(y[-3:]) #shape of sp: (uNum,pNum), shape of I_tot: (pNum)

    delta_ = delta + 2*pi*beta*t #frequency chirping
    # shape of k: (pNum,3)
    # shape of rx = (lNum,uNum,pNum), sp.shape = (pNum) ==> (rx*sp).shape = (lNum,uNum,pNum)
    # R = Gamma/2 * (rx*sp) / ( 1+4*(delta)**2/Gamma**2 )
    R = np.reshape(Gamma,(1,-1,1))/2 * (rx*sp) / ( 1+4*( delta_ - np.dot(k,y[lNum+uNum:lNum+uNum+3]) )**2/np.reshape(Gamma,(1,-1,1))**2 )    
    # sum R over pNum
    R_sum = np.sum(R,axis=2)
    
    # __________ODE:__________
    # N_l' = ...
    for l in range(lNum):
        for u in range(uNum):
            dydt[l] += Gamma[u]* r[l,u] * y[lNum+u] + R_sum[l,u] * (y[lNum+u] - y[l])
    if not M.size == 0:
        for l1 in range(lNum):
            for l2 in range(lNum):
                dydt[l1] -= M[l1,l2] * (y[l1]-y[l2])
    # N_u' = ... 
    for u in range(uNum):
        dydt[lNum+u]  = -Gamma[u]*y[lNum+u]
        for l in range(lNum):
            dydt[lNum+u] += R_sum[l,u] * (y[l] - y[lNum+u])
    # v' = ...
    for i in range(3):
        for l in range(lNum):
            for u in range(uNum):
                for p in range(pNum):
                    dydt[lNum+uNum+i] +=  hbar/m * k[p,i] * R[l,u,p] * ( y[l] - y[lNum+u] )
    # r' = ...    
    dydt[lNum+uNum+3:lNum+uNum+3+3] = y[lNum+uNum:lNum+uNum+3]
              
    return dydt