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

    diff = data_expmt-computational_model(beta_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*n*np.log(2*np.pi*param_sig_Sq) - 0.5*np.sum(diff**2/param_sig_Sq)

def ln_likelihood_simple(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2

    diff = data_expmt-computational_model(beta_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*np.sum(diff**2/param_sig_Sq)

def computational_model(beta, bc_left, bc_right, params):
    return (AdvDiff_solve(beta, params[1], params[0], bc_left, bc_right))

def ln_likelihood_poly_fit(data_experiment):
    Nsamples_f = 10
    Nsamples_k = 15
    samples_f = np.random.uniform(8.0, 12.0, Nsamples_f)
    samples_k = np.random.uniform(0.01,0.5, Nsamples_k)
    N_tot = Nsamples_f*Nsamples_k
    y = np.zeros((N_tot,1), dtype=float)
    A = np.zeros((N_tot,6), dtype=float)
    for i in range(N_tot):
        j2D = np.floor(i/Nsamples_f)
        i2D = i - Nsamples_f*j2D
        i2D = i2D.astype(int)
        j2D = j2D.astype(int)
        # print(i2D,' ',j2D)
        new_params = np.array([samples_f[i2D], samples_k[j2D]])
        y[i,0] = ln_likelihood_simple(data_experiment, new_params)
        A[i,:] = [1, samples_f[i2D], samples_f[i2D]**2, np.log(samples_k[j2D]), samples_k[j2D], samples_f[i2D]*samples_k[j2D]]

    y = np.matmul(np.transpose(A), y)
    A = np.matmul(np.transpose(A), A)
    a = np.linalg.solve(A,y)

    return a

np.random.seed(12345)

n_data = 50

# true source
source_true = 10.0
kappa_true = 0.2
# true noise added to data
sig_true = 3.0

# deterministic parameters
beta_deterministic     = 2.0
bc_left_deterministic  = 0.5
bc_right_deterministic = 0.0

# clean data
data_true = AdvDiff_solve(beta_deterministic, kappa_true, source_true, bc_left_deterministic, bc_right_deterministic)
# random noise
noise_true = np.random.normal(0,sig_true,n_data)
# noisy data
data_experiment = data_true + noise_true


# VB parameters
tau_f = 1.0
mu_f = 14.0
a_0_k = 0.25
b_0_k = 1.0
Exp_k = a_0_k/b_0_k

Niter=50
mean_f = np.zeros([Niter+1,1])
mean_k = np.zeros([Niter+1,1])
mean_f[0] = mu_f
mean_k[0] = a_0_k/b_0_k

A = ln_likelihood_poly_fit(data_experiment)
mu_like = -(A[1]+A[5]*Exp_k)/(2*A[2])
tau_like = -2*A[2]
# print(tau_like)
for k in range(Niter):


    mu_N = (mu_like*tau_like + tau_f*mu_f)/(tau_like + tau_f)
    tau_N = (tau_f + tau_like)/2.0

    tau_f = tau_N
    mu_f = mu_N
    mean_f[k+1] = mu_N
    Exp_f = mu_N
    Exp_f_sq = mu_N**2 + 1.0/tau_N

    a_N_k = a_0_k + A[3]
    b_N_k = b_0_k - A[4] - A[5]*Exp_f
    # b_N = b_0_tau + 0.5*tau_like_tmp*Exp_f_sq - tau_like_tmp*mu_like*Exp_f + 0.5*tau_like_tmp*mu_like**2

    a_0_k = a_N_k
    b_0_k = b_N_k
    mean_k[k+1] = a_N_k/b_N_k
    Exp_k = a_N_k/b_N_k

    print('ITERATION=',k)
    print('f=',[mu_N, tau_N])
    print('k=',[Exp_k, Exp_k/b_N_k])
    print('\n')

plot_each_chain = True
if(plot_each_chain):
    fig,axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(mean_f, color='k', drawstyle='steps')
    axes[1].plot(mean_k, color='k', drawstyle='steps')

    axes[0].axhline(source_true, color='r', label='source_true')
    axes[0].legend(loc='best')
    axes[0].set_ylabel('$\mathbb{E}[f]$')

    axes[1].axhline(kappa_true, color='r', label='kappa_true')
    axes[1].legend(loc='best')
    axes[1].set_ylabel('$\mathbb{E}[\kappa]$')
    plt.savefig('VB_f-k_chains.png')
    plt.show()

plot_chain_in_paramSpace = False
if (plot_chain_in_paramSpace):
    plt.plot(source_true, kappa_true, marker='o', color='r', zorder=10)
    plt.plot(mean_f, mean_k, marker='', color='k', linewidth=1.)

    plt.ylabel('$\mathbb{E}[\kappa]$')
    plt.xlabel('$\mathbb{E}[f]$')
    plt.savefig('VB_f-k_paramspace.png')
    plt.show()

Npts = 250
x1 = np.linspace(6.0, 12.0, Npts)
x2 = np.linspace(0.01, 1.0, Npts)
X1, X2 = np.meshgrid(x1,x2)
Z = 0*X1 #x2 x x1

# print(tau_N)

for i in range(Npts):
    ln_p_kappa = gamma.pdf(x2[i], a_N_k, 0.0, 1.0/b_N_k)
    for j in range(Npts):
        ln_p_source = norm.pdf(x1[j], mu_N, 1.0/tau_N)
        Z[i,j] = ln_p_kappa*ln_p_source

# print(Z)
plt.figure()
plt.contour(X1, X2, Z)
plt.axvline(source_true, color='r', linestyle='--', label='$f_{true}$')
plt.axhline(kappa_true, color='b', linestyle='--', label='$\kappa_{true}$')
plt.legend(loc='best')
plt.title('$P(\\theta)$')
plt.ylabel('$\kappa$')
plt.xlabel('$f$')
plt.savefig('VB_f-k_joint_dist.png')
plt.show()

plt.figure()
p_source = norm.pdf(x1, mu_N, 1.0/tau_N)
plt.plot(x1, p_source)
plt.axvline(source_true, color='r', linestyle='--', label='true')
plt.legend(loc='best')
plt.title('$P(f)$')
plt.ylabel('$P(f)$')
plt.xlabel('$f$')
plt.savefig('VB_f-k_f_dist.png')
plt.show()

plt.figure()
p_kappa = gamma.pdf(x2, a_N_k, 0.0, 1.0/b_N_k)
plt.plot(x2, p_kappa)
plt.axvline(kappa_true, color='r', linestyle='--', label='true')
plt.legend(loc='best')
plt.title('$P(\kappa)$')
plt.ylabel('$P(\kappa)$')
plt.xlabel('$\kappa$')
plt.savefig('VB_f-k_k_dist.png')
plt.show()
