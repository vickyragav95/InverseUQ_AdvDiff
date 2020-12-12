import numpy as np
from scipy import special as sc
from scipy.stats import gamma
from scipy.stats import norm
import matplotlib.pyplot as plt
import corner
import copy
from AdvDiff_solver import AdvDiff_solve
# actual log likelihood terms
def ln_likelihood(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2
    if (len(params)==2):
        param_sig_Sq = params[1]**2

    diff = data_expmt-computational_model(beta_deterministic, source_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*n*np.log(2*np.pi*param_sig_Sq) - 0.5*np.sum(diff**2/param_sig_Sq)

# log-likelihood of only exponential terms in likelihood
def ln_likelihood_simple(data_expmt, params):
    n = len(data_expmt)

    param_sig_Sq = sig_true**2
    if (len(params)==2):
        param_sig_Sq = params[1]**2

    diff = data_expmt-computational_model(beta_deterministic, source_deterministic, bc_left_deterministic, bc_right_deterministic, params)

    return -0.5*np.sum(diff**2/param_sig_Sq)

def computational_model(beta, source, bc_left, bc_right, params):
    return (AdvDiff_solve(beta, params[0], source, bc_left, bc_right))

# fits gamma distribution on simple log likelihood term
def ln_likelihood_func_fit(data_experiment, params):
    Nsamples = 15
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
kappa_true = 0.3
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


# VB initial parameters
Exp_tau_y = 0.25
a_0_kappa = 2.0
b_0_kappa = 2.0
a_0_tau = 2.0
b_0_tau = 3.0
Niter=100
mean_k = np.zeros([Niter+1,1])
mean_tau = np.zeros([Niter+1,1])
mean_k[0] = a_0_kappa/b_0_kappa
mean_tau[0] = a_0_tau/b_0_tau
for k in range(Niter):
    nrml_lkhood_params = np.array([1.0, (1.0/Exp_tau_y)])
    # compute new gamma dist fit for updated Exp_tau_y
    [A_like, B_like, C_like] = ln_likelihood_func_fit(data_experiment, nrml_lkhood_params)
    A_like = A_like.item()
    B_like = B_like.item()
    C_like = C_like.item()


    # compute new a,b params for kappa
    a_N_kappa = a_0_kappa + B_like
    b_N_kappa = b_0_kappa - A_like
    # update priors
    a_0_kappa = a_N_kappa
    b_0_kappa = b_N_kappa
    mean_k[k+1] = a_N_kappa/b_N_kappa
    Exp_kappa = a_N_kappa/b_N_kappa
    Exp_kappa_sq = Exp_kappa**2 + b_N_kappa/Exp_kappa

    A_like_tmp = A_like/Exp_tau_y
    B_like_tmp = B_like/Exp_tau_y
    C_like_tmp = C_like/Exp_tau_y
    # compute new a,b params for tau_y
    a_N_tau = a_0_tau + n_data/2 + 1
    b_N_tau = b_0_tau + A_like_tmp*a_N_kappa/b_N_kappa + C_like_tmp + B_like_tmp*(np.log(b_N_kappa) + sc.digamma(a_N_kappa))
    # update priors
    a_0_tau = a_N_tau
    b_0_tau = b_N_tau
    Exp_tau_y = a_N_tau/b_N_tau
    mean_tau[k+1] = a_N_tau/b_N_tau
    # Exp_tau_y = 1.0/sig_true**2

    print('ITERATION=',k)
    print('k=',[Exp_kappa, Exp_kappa/b_N_kappa])
    print('tau=',[Exp_tau_y, Exp_tau_y/b_N_tau])
    print('\n')


plot_each_chain = True
if(plot_each_chain):
    fig,axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(mean_k, color='k', drawstyle='steps')
    axes[1].plot(mean_tau, color='k', drawstyle='steps')

    axes[0].axhline(kappa_true, color='r', label='kappa_true')
    axes[0].legend(loc='best')
    axes[0].set_ylabel('$\mathbb{E}[\kappa]$')

    axes[1].axhline(1/sig_true**2, color='r', label='tau_true')
    axes[1].legend(loc='best')
    axes[1].set_ylabel('$\mathbb{E}[Tau_y]$')
    plt.savefig('VB_k-sig_chains.png')
    plt.show()

plot_chain_in_paramSpace = False
if (plot_chain_in_paramSpace):
    plt.plot(kappa_true, 1/sig_true**2, marker='o', color='r', zorder=10)
    plt.plot(mean_k, mean_tau, marker='', color='k', linewidth=1.)

    plt.ylabel('$\mathbb{E}[Tau_y]$')
    plt.xlabel('$\mathbb{E}[\kappa]$')
    plt.savefig('VB_f-sig_paramspace.png')
    plt.show()

Npts = 250
x1 = np.linspace(0.01, 0.5, Npts)
x2 = np.linspace(0.01, 0.5, Npts)
X1, X2 = np.meshgrid(x1,x2)
Z = 0*X1 #50x100
# print(np.shape(Z))
# print(a_N_tau)
# print(a_N_kappa)

for i in range(Npts):
    ln_p_tau = gamma.pdf(x2[i], a_N_tau, 0.0, 1.0/b_N_tau)
    for j in range(Npts):
        ln_p_kappa = gamma.pdf(x1[j], a_N_kappa, 0.0, 1.0/b_N_kappa)
        Z[i,j] = ln_p_tau*ln_p_kappa

# print(Z)
plt.figure()
plt.contour(X1, X2, Z)
plt.axvline(kappa_true, color='r', linestyle='--', label='$\kappa_{true}$')
plt.axhline(1/sig_true**2, color='b', linestyle='--', label='$\\tau_{y,true}$')
plt.legend(loc='best')
plt.title('$P(\\theta)$')
plt.ylabel('$\\tau_y$')
plt.xlabel('$\kappa$')
plt.savefig('VB_k-sig_joint_dist.png')
plt.show()

plt.figure()
p_kappa = gamma.pdf(x1, a_N_kappa, 0.0, 1.0/b_N_kappa)
plt.plot(x1, p_kappa)
plt.axvline(kappa_true, color='r', linestyle='--', label='true')
plt.legend(loc='best')
plt.title('$P(\kappa)$')
plt.ylabel('$P(\kappa)$')
plt.xlabel('$\kappa$')
plt.savefig('VB_k-sig_k_dist.png')
plt.show()

plt.figure()
p_tau = gamma.pdf(x2, a_N_tau, 0.0, 1.0/b_N_tau)
plt.plot(x2, p_tau)
plt.axvline(1/sig_true**2, color='r', linestyle='--', label='true')
plt.legend(loc='best')
plt.title('$P(\\tau_y)$')
plt.ylabel('$P(\\tau_y)$')
plt.xlabel('$\\tau_y$')
plt.savefig('VB_k-sig_sig_dist.png')
plt.show()

fig, axs = plt.subplots(2,2)
axs[0,0].plot(x1,p_kappa)
axs[0,0].axvline(kappa_true, color='r', linestyle='--', label='true')
axs[0,0].legend(loc='best')
axs[0,0].set_title('$P(\kappa)$')
axs[0,0].set_ylabel('$P(\kappa)$')

axs[1,0].contour(X1, X2, Z)
axs[1,0].axvline(kappa_true, color='r', linestyle='--', label='$\kappa_{true}$')
axs[1,0].axhline(1/sig_true**2, color='b', linestyle='--', label='$\\tau_{y,true}$')
axs[1,0].legend(loc='best')
axs[1,0].set_ylabel('$\\tau_y$')
axs[1,0].set_xlabel('$\kappa$')

axs[1,1].plot(x2, p_tau)
axs[1,1].axvline(1/sig_true**2, color='r', linestyle='--', label='true')
axs[1,1].legend(loc='best')
axs[1,1].set_title('$P(\\tau_y)$')
axs[1,1].set_xlabel('$\\tau_y$')

axs[0,1].axis('off')

plt.savefig('k-sig_MargPDFs.pdf')
plt.show()
