import numpy as np
import matplotlib.pyplot as plt
import corner
import copy
from AdvDiff_solver import AdvDiff_solve

def ln_likelihood(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2
    if (len(params)==2):
        param_sig_Sq = params[1]**2

    diff = data_expmt-computational_model(beta_deterministic, source_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*n*np.log(2*np.pi*param_sig_Sq) - 0.5*np.sum(diff**2/param_sig_Sq)

def ln_likelihood_simple(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2
    if (len(params)==2):
        param_sig_Sq = params[1]**2

    diff = data_expmt-computational_model(beta_deterministic, source_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*np.sum(diff**2/param_sig_Sq)


def computational_model(beta, source, bc_left, bc_right, params):
    return (AdvDiff_solve(beta, params[0], source, bc_left, bc_right))


def ln_likelihood_poly_fit(data_experiment, params):
    Nsamples = 20
    samples = np.random.uniform(0.0, 1.0, Nsamples)
    y = np.zeros((Nsamples,1), dtype=float)
    A = np.zeros((Nsamples,3), dtype=float)
    for i in range(Nsamples):
        if (len(params)==2):
            new_params = [samples[i], params[1]]
        else:
            new_params = [samples[i]]
        y[i,0] = ln_likelihood_simple(data_experiment, new_params)
        A[i,:] = [samples[i], np.log(samples[i]), 1.0]

    y = np.matmul(np.transpose(A), y)
    A = np.matmul(np.transpose(A), A)
    a = np.linalg.solve(A,y)

    return a[0], a[1], a[2]

np.random.seed(12345)

n_data = 50

# true source
kappa_true = 0.1
# true noise added to data
sig_true = 2.0

# deterministic parameters
# kappa_deterministic    = 0.1
source_deterministic   = 10.0
beta_deterministic     = 2.0
bc_left_deterministic  = 0.5
bc_right_deterministic = 0.0

# clean data
data_true = AdvDiff_solve(beta_deterministic, kappa_true, source_deterministic, bc_left_deterministic, bc_right_deterministic)
# random noise
noise_true = np.random.normal(0,sig_true,n_data)
# noisy data
data_experiment = data_true + noise_true


params0 = np.linspace(0.01,1.0, num=100)
params0 = np.reshape(params0, (100,1))
new_ln_like = np.zeros((100,1), dtype=float)
for i in range(100):
    new_params = [params0[i,0]]
    new_ln_like[i,0] = ln_likelihood_simple(data_experiment, new_params)


[A, B, C] = ln_likelihood_poly_fit(data_experiment, np.array([0.1]))
A = A.item()#np.asscalar(new_mu)
B = B.item()#np.asscalar(new_sig)
C = C.item()#np.asscalar(new_off)
# print(new_mu)
# print(new_sig)
# print(new_off)

new_approx = np.zeros((100,1), dtype=float)
for i in range(100):
    new_approx[i,0] = B*np.log(params0[i,0]) + A*params0[i,0] + C

# plt.plot(params0,np.exp(new_ln_like))
# plt.plot(params0,new_ln_like)
plt.figure()
plt.plot(params0,new_approx,'r--',params0,new_ln_like,'b-')
plt.show()

plt.figure()
plt.plot(params0,np.exp(new_approx),'r--',params0,np.exp(new_ln_like),'b-')
plt.show()
