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

np.random.seed(12345)

n_data = 50

# true source
source_true = 10.0
# true noise added to data
sig_true = 2.0

# deterministic parameters
kappa_deterministic    = 0.2
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
a_0_tau = 1.0
b_0_tau = 0.5
Niter=100
mean_f = np.zeros([Niter+1,1])
mean_tau = np.zeros([Niter+1,1])
mean_f[0] = mu_f
mean_tau[0] = a_0_tau/b_0_tau
for k in range(Niter):
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

    mean_f[k+1] = mu_N

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

    mean_tau[k+1] = a_N_tau/b_N_tau

    print('ITERATION=',k)
    print('f=',[mu_N, tau_N])
    print('tau=',[Exp_tau_y, Exp_tau_y/b_N_tau])
    print('\n')


plot_each_chain = True
if(plot_each_chain):
    fig,axes = plt.subplots(2, 1, sharex=True)

    axes[0].plot(mean_f, color='k', drawstyle='steps')
    axes[1].plot(mean_tau, color='k', drawstyle='steps')

    axes[0].axhline(source_true, color='r', label='source_true')
    axes[0].legend(loc='best')
    axes[0].set_ylabel('$\mathbb{E}[f]$')

    axes[1].axhline(1/sig_true**2, color='r', label='tau_true')
    axes[1].legend(loc='best')
    axes[1].set_ylabel('$\mathbb{E}[Tau_y]$')
    plt.savefig('VB_f-sig_chains.png')
    plt.show()

plot_chain_in_paramSpace = False
if (plot_chain_in_paramSpace):
    plt.plot(source_true, 1/sig_true**2, marker='o', color='r', zorder=10)
    plt.plot(mean_f, mean_tau, marker='', color='k', linewidth=1.)

    plt.ylabel('$\mathbb{E}[Tau_y]$')
    plt.xlabel('$\mathbb{E}[f]$')
    plt.savefig('VB_f-sig_paramspace.png')
    plt.show()

Npts = 250
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
plt.axvline(source_true, color='r', linestyle='--', label='$f_{true}$')
plt.axhline(1/sig_true**2, color='b', linestyle='--', label='$\\tau_{y,true}$')
plt.legend(loc='best')
plt.title('$P(\\theta)$')
plt.ylabel('$\\tau_y$')
plt.xlabel('$f$')
plt.savefig('VB_f-sig_joint_dist.png', dpi=300)
plt.show()

plt.figure()
p_source = norm.pdf(x1, mu_N, 1.0/tau_N)
plt.plot(x1, p_source)
plt.axvline(source_true, color='r', linestyle='--', label='true')
plt.legend(loc='best')
plt.title('$P(f)$')
plt.ylabel('$P(f)$')
plt.xlabel('$f$')
plt.savefig('VB_f-sig_f_dist.png', dpi=300)
plt.show()

plt.figure()
p_tau = gamma.pdf(x2, a_N_tau, 0.0, 1.0/b_N_tau)
plt.plot(x2, p_tau)
plt.axvline(1/sig_true**2, color='r', linestyle='--', label='true')
plt.legend(loc='best')
plt.title('$P(\\tau_y)$')
plt.ylabel('$P(\\tau_y)$')
plt.xlabel('$\\tau_y$')
plt.savefig('VB_f-sig_sig_dist.png', dpi=300)
plt.show()

fig, axs = plt.subplots(2,2)
axs[0,0].plot(x1,p_source)
axs[0,0].axvline(source_true, color='r', linestyle='--', label='true')
axs[0,0].legend(loc='best')
axs[0,0].set_title('$P(f)$')
axs[0,0].set_ylabel('$P(f)$')
axs[0,0].set_xlabel('$f$')

axs[1,0].contour(X1, X2, Z)
axs[1,0].axvline(source_true, color='r', linestyle='--', label='$f_{true}$')
axs[1,0].axhline(1/sig_true**2, color='b', linestyle='--', label='$\\tau_{y,true}$')
axs[1,0].legend(loc='best')
axs[1,0].set_ylabel('$\\tau_y$')
axs[1,0].set_xlabel('$f$')

axs[1,1].plot(x2, p_tau)
axs[1,1].axvline(1/sig_true**2, color='r', linestyle='--', label='true')
axs[1,1].legend(loc='best')
axs[1,1].set_title('$P(\\tau_y)$')
axs[1,1].set_ylabel('$P(\\tau_y)$')
axs[1,1].set_xlabel('$\\tau_y$')

axs[0,1].axis('off')

plt.savefig('f-sig_MargPDFs.pdf')
plt.show()
