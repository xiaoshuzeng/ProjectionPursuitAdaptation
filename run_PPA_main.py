#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from _projection_pursuit_adaptation_tools import *
%matplotlib qt5


# %%
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
paper_figure = True
if paper_figure:
    plt.rc('font', size=14)    # for paper
else:
    plt.rc('font', size=16)    # for presentation


# %%
#######################
# Input specifications
#######################
main_verbose = 1
nord = 3   # order of PCE. This parameter is application dependent
ndim = 7   # dimension
pc_type = "HG"  # Polynomial type
n_MC = 100000   # #of MC samples of generate KDE of the PDF

# %%
#####################
# load training data
#####################
# the input must be Gaussian. If not, please transform to Gaussian first
training = np.loadtxt('training.txt')
xi = training[:, :-1]
Q_evals = training[:, -1]

xpts_MC_data, PDF_data_MC_data = KDE(Q_evals)
dxpts = xpts_MC_data[1] - xpts_MC_data[0]
CDF_data_MC_data = np.cumsum(dxpts*PDF_data_MC_data)

N_data = xi.shape[0]


# %%
##############################
# ProjectionPursuitAdaptation
##############################
tol_pce = 2e-2    # this parameter is application dependent

PPA_dim = None
# If not None, then use the specified dimension for adaptation.
# If None, then find the optimal adapted dimension based on tol_pce

PPA_method = 1  # defaut is 1, which uses PPA method
recover_run = True
if not recover_run:
    # fresh run starting from dimension 1
    PPA_model = ProjectionPursuitAdaptation(tol_pce, PPA_method=PPA_method, PPA_dim=PPA_dim, main_verbose=main_verbose)
    PPA_model.projection_pursuit_adaptation(nord, pc_type, xi, Q_evals)
    ppa_dim, list_vec_a, mat_A_new, list_c_k, list_pce_evals = PPA_model._ndim_iteration, PPA_model._list_vec_a, PPA_model._mat_A_new, PPA_model._list_c_k, PPA_model._list_pce_evals
else:
    # recover run with given starting dimension
    PPA_dim = 5
    PPA_model = ProjectionPursuitAdaptation(tol_pce, PPA_method=PPA_method, PPA_dim=PPA_dim, recover_run=True, ndim_iteration=ppa_dim,
                                            list_vec_a=list_vec_a, mat_A_new=mat_A_new, list_c_k=list_c_k, list_pce_evals=list_pce_evals, main_verbose=main_verbose)
    PPA_model.projection_pursuit_adaptation(nord, pc_type, xi, Q_evals)
    ppa_dim, list_vec_a, mat_A_new, list_c_k, list_pce_evals = PPA_model._ndim_iteration, PPA_model._list_vec_a, PPA_model._mat_A_new, PPA_model._list_c_k, PPA_model._list_pce_evals


# %%
#####################################################
# generate MC samples from the PPA model to plot PDF
#####################################################
if pc_type == 'HG':
    germ_samples = np.random.normal(0, 1, (n_MC, ppa_dim))
elif pc_type == 'LU':
    germ_samples = np.random.uniform(-1, 1, (n_MC, ppa_dim))

if PPA_method == 0:
    pce_evals = PPA_model.evaluation(germ_samples)
elif PPA_method == 1:
    pce_evals = PPA_model.evaluation(germ_samples)

xpts_MC_pp_adapt, PDF_data_MC_pp_adapt = KDE(pce_evals)
dxpts = xpts_MC_pp_adapt[1] - xpts_MC_pp_adapt[0]
CDF_data_MC_pp_adapt = np.cumsum(dxpts*PDF_data_MC_pp_adapt)


# %%
#######################################
# PDF data obtained from reference MC
#######################################
Q_evals_ref = np.loadtxt('reference.txt')
xpts_MC_ref, PDF_data_MC_ref = KDE(Q_evals_ref)
dxpts = xpts_MC_ref[1] - xpts_MC_ref[0]
CDF_data_MC_ref = np.cumsum(dxpts*PDF_data_MC_ref)


# %%
###########################
# for comparison purposes
###########################
def interpolate_on_ref(xpts, PDF_data, xpts_MC_ref):
    return np.interp(xpts_MC_ref, xpts, PDF_data)


PDF_data_MC_pp_adapt_interp1 = interpolate_on_ref(
    xpts_MC_pp_adapt, PDF_data_MC_pp_adapt, xpts_MC_ref)
rela_2norm1 = np.linalg.norm(PDF_data_MC_pp_adapt_interp1 -
                             PDF_data_MC_ref)/np.linalg.norm(PDF_data_MC_ref)


PDF_data_MC_pp_adapt_interp0 = interpolate_on_ref(
    xpts_MC_pp_adapt, PDF_data_MC_pp_adapt, xpts_MC_ref)
rela_2norm0 = np.linalg.norm(PDF_data_MC_pp_adapt_interp0 -
                             PDF_data_MC_ref)/np.linalg.norm(PDF_data_MC_ref)


PDF_data_MC_pp_adapt_interp_d = interpolate_on_ref(
    xpts_MC_data, PDF_data_MC_data, xpts_MC_ref)
rela_2norm_d = np.linalg.norm(PDF_data_MC_pp_adapt_interp_d -
                              PDF_data_MC_ref)/np.linalg.norm(PDF_data_MC_ref)


# %%
############
# test data
############
test = np.loadtxt('test.txt')
xi_test = test[:, :-1]
Q_evals_test = test[:, -1]
pce_evals_test = PPA_model.evaluation(xi_test)


# %%
#######################
# plots of comparisons
#######################
plt.figure()
plt.plot(xpts_MC_ref, PDF_data_MC_ref, '-', label='Ref')
plt.plot(xpts_MC_data, PDF_data_MC_data, '--', label='Data (%d)' % (N_data))
plt.plot(xpts_MC_pp_adapt, PDF_data_MC_pp_adapt, '-.', label='%dd PPA' % ppa_dim)
plt.legend(loc=1)
plt.xlabel(r'Water flow rate ($m^3/yr$)')
plt.ylabel('PDF')
plt.xlim([20, 140])
plt.title(r'Relative $l^2$ err of data and PPA are %.3f and %.3f' % (rela_2norm_d, rela_2norm1))
plt.tight_layout()

plt.figure()
plt.plot(xpts_MC_ref, CDF_data_MC_ref, '-', label='Ref')
plt.plot(xpts_MC_data, CDF_data_MC_data, '--', label='Data (%d MC)' % (N_data))
plt.plot(xpts_MC_pp_adapt, CDF_data_MC_pp_adapt, '-.', label='%dd PPA' % ppa_dim)
plt.legend(loc=4)
plt.xlabel(r'Water flow rate ($m^3/yr$)')
plt.ylabel('CDF')
plt.xlim([20, 140])
plt.tight_layout()

plt.figure()
plt.plot(Q_evals_test, 'o', label='Ref')
plt.plot(pce_evals_test, 'o', label='PPA')
plt.xlabel('Label of test data')
plt.ylabel('Test data')
plt.legend()
plt.grid(linestyle=':')
rela_l2 = np.linalg.norm(pce_evals_test-Q_evals_test)/np.linalg.norm(Q_evals_test)
plt.title(r'Test set of PPA (relative $l^2$ err is %.4f)' % rela_l2)
plt.tight_layout()

plt.figure()
plt.plot(Q_evals_test, pce_evals_test, 'o')
plt.xlabel('Test data')
plt.ylabel('Prediction')
minx = np.min([np.min(Q_evals_test), np.min(pce_evals_test)])
maxx = np.max([np.max(Q_evals_test), np.max(pce_evals_test)])
Lx = maxx - minx
s = 0.05
plt.plot([minx-s*Lx, maxx+s*Lx], [minx-s*Lx, maxx+s*Lx])
plt.grid(linestyle=':')
rela_l2 = np.linalg.norm(pce_evals_test-Q_evals_test)/np.linalg.norm(Q_evals_test)
plt.title(r'Test vs pred (relative $l^2$ err is %.4f)' % rela_l2)
plt.tight_layout()


# %%
##
