import os
import sys
import pickle
import pathlib
import numpy as np
import pandas as pd
import pprint as pp
import pysindy as ps
from scipy import fft
from pathlib import Path
from scipy.integrate import solve_ivp
from sklearn.preprocessing import MinMaxScaler

# Ignore matplotlib deprecation warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Seed the random number generators for reproducibility
np.random.seed(100)

# Update path to include mypkg
sys.path.insert(0, str(Path(__file__).parent.parent.absolute()))

from src import helpers, plot_data, global_config, datasets
config = global_config.config

# ## Load data (and smooth it)

# Hyperparameters

# Initial condition parameters
n_avg = 1 # number of curves for moving average
u_true_cutoff = 150 # final index for MNIST propogated wave
u_true_start = 10

# IVP parameters
dx = None
x = None
t = None
dt = None

# Load the data
dataset = "MNIST"
output_dir = pathlib.Path(config.top_dir) / "fft_images" / dataset

file = output_dir / "pkl" / f"{dataset.lower()}_og_integral.pkl"
with open(file, 'rb') as file:
    d = pickle.load(file)

# Calculate moving average
u_total = helpers.moving_average(d['ints'],n=n_avg,axis=0)
u_true = u_total[u_true_start:u_true_cutoff, :]
x = np.asarray(d['r_x'])
t_total, t_true = (np.arange(0,u_total.shape[0]) + 0., np.arange(0,u_true.shape[0]) + 0.)
dt = 1.

x.min()
x_scaled, x_min, x_max = helpers.min_max_fit(x, 0., 1.)
t_scaled, t_min, t_max = helpers.min_max_fit(t_true, 0., 1.)
u_scaled, u_min, u_max = helpers.min_max_fit(u_true, 0., 1.)

# ## Feature Library

u = u_scaled; x = x_scaled; t = t_scaled;
dx = x[1]-x[0]; dt = t[1]-t[0]

dummy_u = np.random.randn(x.shape[0], t.shape[0], 1)

# Create unique 2 value combinations of [1,2,6,7]
from itertools import combinations
select_features = [1,2,6,7]
feature_indices_list = list(combinations(select_features, 2))

# Add full set of features
feature_indices_list.append(select_features)

for ensemble_idx, feature_indices in enumerate(feature_indices_list):

    # Define PDE library that is quadratic in u, and
    # third-order in spatial derivatives of u.
    if 0:
        pde_lib = ps.PDELibrary(function_library=ps.PolynomialLibrary(degree=2,include_bias=False),
                                derivative_order=3, spatial_grid=x,
                                include_bias=True, is_uniform=True)

    else:
        library_functions = [lambda x: x, lambda x: x**2]
        function_names = [lambda x: f"{x}", lambda x: f"{x}^2"]
        multiindices = [[1],[2],[3]]
        #feature_indices = [1,2,6,7] # [1,2,6,7]

        if 0:
            multiindices = None
            feature_indices = None
        
        pde_lib = ps.PDELibrary(function_library=ps.CustomLibrary(library_functions=library_functions,function_names=function_names,include_bias=False),
                                derivative_order=3, spatial_grid=x,
                                include_bias=True, is_uniform=True, multiindices = multiindices, feature_indices=feature_indices)

    dummy_pde_lib = pde_lib
    dummy_pde_lib.fit([dummy_u])
    feature_names = [helpers.modify_pde_sindy_out(feature) for feature in dummy_pde_lib.get_feature_names()]
    print("Library:")
    print(feature_names)

    # ## Prepare input

    # Convert to PySINDy format
    if u_scaled.shape[1] != len(t_scaled):
        u_scaled = u_scaled.T
    if u_scaled.shape[1] != len(t_scaled):
        raise Exception(f"u shape is weird: {u.shape}, x shape {x.shape}, t shape {t.shape}")
    u_in = np.expand_dims(u_scaled, -1)
    dt = t_scaled[1] - t_scaled[0]

    # Split into train and test
    train_split = 1.0
    u_train = u_in[:, :int(train_split*u_in.shape[1]), :]
    u_test = u_in[:, int(train_split*u_in.shape[1]):, :]

    print(u_scaled.shape)
    print(x_scaled.shape)
    print(t_scaled.shape)
    print(u_train.shape)
    print(u_test.shape)

    # ## Ensemble fit

    ensemble_optimizer = ps.EnsembleOptimizer(
        ps.STLSQ(threshold=20, alpha=1e-5, normalize_columns=True),
        bagging=True,
        n_subset=int(0.9 * u_train.shape[0] * u_train.shape[1]),
        n_models=10000
    )

    model = ps.SINDy(optimizer=ensemble_optimizer, feature_names=feature_names, feature_library=pde_lib)
    model.fit(u_train, t=dt)
    ensemble_scores = np.asarray(ensemble_optimizer.score_list).squeeze()
    ensemble_coefs = np.asarray(ensemble_optimizer.coef_list).squeeze()
    mean_ensemble = np.mean(ensemble_coefs, axis=0)
    std_ensemble = np.std(ensemble_coefs, axis=0)

    # Collect scores for full set
    full_scores = list()
    for i in range(ensemble_coefs.shape[0]):
        model.optimizer.coef_ = ensemble_coefs[i][np.newaxis,:]
        full_scores.append(model.score(u_train))
    full_scores = np.asarray(full_scores)

    if 1:
        # Filter for 75th percentile
        percentile_75_1 = pd.DataFrame(ensemble_scores).describe().loc['75%'].item()
        percentile_75_2 = pd.DataFrame(full_scores).describe().loc['75%'].item()
        ensemble_coefs = ensemble_coefs[(ensemble_scores >= percentile_75_1) & (full_scores >= percentile_75_2), :]
        ensemble_scores = ensemble_scores[(ensemble_scores >= percentile_75_1) & (full_scores >= percentile_75_2)]
    else:
        # Don't filter
        pass

    print("coefs shape:", ensemble_coefs.shape)
    print("num features:", len(model.get_feature_names()))
    print("mean:", mean_ensemble.shape)
    print("std:", std_ensemble.shape)

    df = pd.DataFrame(ensemble_coefs)
    df.describe()

    config.pdf_dir = str(Path(__file__).parent)
    plot_data.plot_ensemble(feature_names, mean_ensemble, std_ensemble, title="Mean and Std of Ensemble", xaxis_title="Feature", yaxis_title="Coefficient Value", fname=f"mnist_ensemble_{ensemble_idx}", save=True)

    print()
