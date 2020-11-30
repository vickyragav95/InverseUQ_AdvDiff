import numpy as np
from scipy import special as sc
from scipy.stats import gamma
from scipy.stats import norm
import matplotlib.pyplot as plt
import corner
import copy
from AdvDiff_solver import AdvDiff_solve

def ln_likelihood(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2
    if (len(params)==2):
        param_sig_Sq = params[1]**2

    diff = data_expmt-computational_model(beta_deterministic, kappa_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*n*np.log(2*np.pi*param_sig_Sq) - 0.5*np.sum(diff**2/param_sig_Sq)

def ln_likelihood_simple(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2
    if (len(params)==2):
        param_sig_Sq = params[1]**2

    diff = data_expmt-computational_model(beta_deterministic, kappa_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*np.sum(diff**2/param_sig_Sq)

def computational_model(beta, kappa, bc_left, bc_right, params):
    return (AdvDiff_solve(beta, kappa, params[0], bc_left, bc_right))

def ln_likelihood_poly_fit(data_experiment, params):
    Nsamples = 5
    samples = np.random.uniform(0.0, 20.0, Nsamples)
    y = np.zeros((Nsamples,1), dtype=float)
    A = np.zeros((Nsamples,3), dtype=float)
    for i in range(Nsamples):
        if (len(params)==2):
            new_params = [samples[i], params[1]]
        else:
            new_params = [samples[i]]
        y[i,0] = ln_likelihood_simple(data_experiment, new_params)
        A[i,:] = [1, samples[i], samples[i]**2]

    y = np.matmul(np.transpose(A), y)
    A = np.matmul(np.transpose(A), A)
    a = np.linalg.solve(A,y)

    return -a[1]/(2.0*a[2]), 2.0*np.abs(a[2]), a[0]-a[1]*a[1]/(4.0*a[2])

# def ln_Fpost_poly_fit(data_experiment, params, mu_f, sigsq_f):
#     Nsamples = 10
#     samples = np.random.uniform(0.0, 20.0, Nsamples)
#     y = np.zeros((Nsamples,1), dtype=float)
#     A = np.zeros((Nsamples,3), dtype=float)
#     for i in range(Nsamples):
#         if (len(params)==2):
#             new_params = [samples[i], params[1]]
#         else:
#             new_params = [samples[i]]
#         y[i,0] = ln_likelihood_simple(data_experiment, new_params)
#         y[i,0] += -0.5*((samples[i]-mu_f)**2)/sigsq_f
#         A[i,:] = [1, samples[i], samples[i]**2]
#
#     y = np.matmul(np.transpose(A), y)
#     A = np.matmul(np.transpose(A), A)
#     a = np.linalg.solve(A,y)
#
#     return -a[1]/(2.0*a[2]), 2.0*np.abs(a[2]), a[0]-a[1]*a[1]/(4.0*a[2])
#
# def ln_likelihood_MC(data_experiment, params, mu_f, sigsq_f):
#     Nsamples = 100
#     samples = np.random.normal(mu_f, np.sqrt(sigsq_f), Nsamples)
#     y = np.zeros((Nsamples,1), dtype=float)
#     for i in range(Nsamples):
#         if (len(params)==2):
#             new_params = [samples[i], params[1]]
#         else:
#             new_params = [samples[i]]
#         y[i,0] = ln_likelihood_simple(data_experiment, new_params)
#         y[i,0] += -0.5*((samples[i]-mu_f)**2)/sigsq_f
#
#     y = np.exp(y)
#     mu_like = np.mean(y)
#     tau_like = 1.0/(np.var(y))
#
#     return mu_like, tau_like

np.random.seed(12345)

n_data = 50

# true source
source_true = 10.0
# true noise added to data
sig_true = 1.0

# deterministic parameters
kappa_deterministic    = 0.1
beta_deterministic     = 2.0
bc_left_deterministic  = 0.5
bc_right_deterministic = 0.0

# clean data
data_true = AdvDiff_solve(beta_deterministic, kappa_deterministic, source_true, bc_left_deterministic, bc_right_deterministic)
# random noise
noise_true = np.random.normal(0,sig_true,n_data)
# noisy data
data_experiment = data_true + noise_true


# VB parameters
Exp_tau_y = 5.0
tau_f = 2.0
mu_f = 14.0
a_0_tau = 2.0
b_0_tau = 0.25

for k in range(100):
    nrml_lkhood_params = np.array([1.0, 1.0/Exp_tau_y])
    [mu_like, tau_like, off_like] = ln_likelihood_poly_fit(data_experiment, nrml_lkhood_params)
    # [mu_like, tau_like] = ln_likelihood_MC(data_experiment, nrml_lkhood_params, mu_f, 1.0/tau_f)
    mu_like = mu_like.item()
    tau_like = tau_like.item()
    off_like = off_like.item()

    mu_N = (mu_like*tau_like + tau_f*mu_f)/(tau_like + tau_f)
    tau_N = (tau_f + tau_like)/2.0

    tau_f = tau_N
    mu_f = mu_N

    Exp_f = mu_N
    Exp_f_sq = mu_N**2 + 1.0/tau_N

    tau_like_tmp = tau_like/Exp_tau_y
    off_like_tmp = off_like/Exp_tau_y

    a_N_tau = a_0_tau + n_data/2 + 1
    b_N_tau = b_0_tau + 0.5*tau_like_tmp*Exp_f_sq - tau_like_tmp*mu_like*Exp_f + 0.5*tau_like_tmp*mu_like**2 - off_like_tmp
    # b_N = b_0_tau + 0.5*tau_like_tmp*Exp_f_sq - tau_like_tmp*mu_like*Exp_f + 0.5*tau_like_tmp*mu_like**2

    a_0_tau = a_N_tau
    b_0_tau = b_N_tau
    Exp_tau_y = a_N_tau/b_N_tau

    print('ITERATION=',k)
    print(mu_N)
    print(Exp_tau_y)
    print('\n')

Npts = 500
x1 = np.linspace(6.0, 14.0, Npts)
x2 = np.linspace(0.01, 2.0, Npts)
X1, X2 = np.meshgrid(x1,x2)
Z = 0*X1 #x2 x x1


for i in range(Npts):
    ln_p_tau = gamma.pdf(x2[i], a_N_tau, 0.0, 1.0/b_N_tau)
    for j in range(Npts):
        ln_p_source = norm.pdf(x1[j], mu_N, 1.0/tau_N)
        Z[i,j] = ln_p_tau*ln_p_source

# print(Z)
plt.figure()
plt.contour(X1, X2, Z)
plt.axvline(source_true, color='r', linestyle='--', label='true')
plt.axhline(1/sig_true**2, color='b', linestyle='--', label='true')
plt.show()

plt.figure()
p_source = norm.pdf(x1, mu_N, 1.0/tau_N)
plt.plot(x1, p_source)
plt.axvline(source_true, color='r', linestyle='--', label='true')
plt.show()

plt.figure()
p_tau = gamma.pdf(x2, a_N_tau, 0.0, 1.0/b_N_tau)
plt.plot(x2, p_tau)
plt.axvline(1/sig_true**2, color='r', linestyle='--', label='true')
plt.show()