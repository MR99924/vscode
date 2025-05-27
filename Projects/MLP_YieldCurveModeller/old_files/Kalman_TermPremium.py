import numpy as np
from pykalman import KalmanFilter

# ------------------------------------------------------
# 1) PREPARE DATA (placeholders for illustration)
# ------------------------------------------------------
# Suppose we have T monthly observations of:
# - y_short: short Treasury yield (e.g. 3-month)
# - y_long: long Treasury yield (e.g. 10-year)
# - infl: inflation or inflation gap
# - unemp: unemployment gap
#
# In reality, you'd include more maturities and more macro series.
# We'll keep it minimal for clarity. Shape: (T, )
T = 200  # number of time points for illustration
np.random.seed(42)
y_short = 0.02 + 0.002 * np.random.randn(T)  # ~2% average short yield
y_long = 0.03 + 0.0025 * np.random.randn(T)  # ~3% average long yield
infl = 0.02 + 0.0015 * np.random.randn(T)
unemp = 0.005 * np.random.randn(T)

# Stack the observed variables into a measurement vector:
# Here, we have 4 observables at each time step.
observations = np.column_stack([y_short, y_long, infl, unemp])  # shape (T, 4)

# ------------------------------------------------------
# 2) SPECIFY STATE-SPACE DIMENSIONS
# ------------------------------------------------------
# We'll define a stylized state vector X_t with 5 components:
# 1) cyclical_level (L^C) ~ the short-run deviation in the level factor
# 2) slope (S) ~ slope factor
# 3) infl_gap (Π^C) ~ cyclical inflation gap
# 4) unemp_gap (U^C) ~ cyclical unemployment gap
# 5) trend_rate (L^T) ~ a unit-root process capturing long-run level
#
# This is a simplified demonstration. The actual paper has more structure
# (e.g., separate long-run inflation and real-rate components). But the idea
# is similar: some factors are mean-reverting; others follow unit roots.
n_state_vars = 5
n_obs = observations.shape[1]

# ------------------------------------------------------
# 3) BUILD TRANSITION MATRIX (A) & OTHER DYNAMICS
# ------------------------------------------------------
# A: how states evolve from t-1 to t (X_t = A * X_{t-1} + ...).
# We'll impose block structure:
# - cyclical_level, slope, infl_gap, unemp_gap are mean-reverting
# - trend_rate is a unit root
#
# e.g., a very simplistic approach (diagonal-block style):
A = np.eye(n_state_vars)

# a) partial mean reversion for cyclical factors:
rho_level = 0.95  # cyclical factor persistence
rho_slope = 0.90
rho_infl_gap = 0.85
rho_unemp = 0.90
A[0, 0] = rho_level
A[1, 1] = rho_slope
A[2, 2] = rho_infl_gap
A[3, 3] = rho_unemp

# b) unit root for trend_rate => A[4,4] = 1.0 (already set by np.eye)

# Transition offset (c) could be zero if we assume no drift
c = np.zeros(n_state_vars)

# Process noise covariance (Q): how much random shock each factor gets
Q = np.diag([1e-5, 1e-5, 1e-5, 1e-5, 1e-6])

# ------------------------------------------------------
# 4) BUILD MEASUREMENT MATRIX (H) & NOISE COVARIANCE (R)
# ------------------------------------------------------
# We specify how the observed yields & macro variables map to the states.
#
# Observations(t) = H * X_t + measurement_noise
#
# For instance, let:
# short yield = trend_rate + cyclical_level + small error
# long yield = trend_rate + term premium (which we’ll define below)
# infl = infl_gap + baseline_infl (but we might fold that into trend_rate)
# unemp = unemp_gap
#
# This is a toy version (the actual paper uses Nelson-Siegel loadings and
# separate inflation trend vs. real-rate trend). Also, the “term premium”
# we’ll see is (y_long - average future short rate). For illustration,
# we’ll let the model treat (y_long) as trend_rate + cyclical_level + slope
# etc. You can expand for more maturities and curvature factors.
H = np.zeros((n_obs, n_state_vars))

# short yield ~ L^T + cyclical_level
H[0, 0] = 1.0  # cyclical_level
H[0, 4] = 1.0  # trend_rate

# long yield ~ L^T + cyclical_level + slope
# (In a real model, you'd incorporate factor loadings carefully.)
H[1, 0] = 1.0  # cyclical_level
H[1, 1] = 1.0  # slope factor
H[1, 4] = 1.0  # trend_rate

# inflation ~ infl_gap
H[2, 2] = 1.0

# unemployment gap ~ unemp_gap
H[3, 3] = 1.0

# Measurement noise
R = np.diag([1e-5, 1e-5, 1e-5, 1e-5])

# ------------------------------------------------------
# 5) INITIALIZE THE KALMAN FILTER
# ------------------------------------------------------
kf = KalmanFilter(
    transition_matrices=A,
    transition_offsets=c,
    observation_matrices=H,
    observation_offsets=None,
    transition_covariance=Q,
    observation_covariance=R,
    n_dim_state=n_state_vars,
    n_dim_obs=n_obs,
    initial_state_mean=np.zeros(n_state_vars),
    initial_state_covariance=1e-2 * np.eye(n_state_vars)
)

# ------------------------------------------------------
# 6) RUN THE FILTER ON THE DATA
# ------------------------------------------------------
state_means, state_covs = kf.filter(observations)
# The 'state_means' array has shape (T, n_state_vars)
# Columns: [cyclical_level, slope, infl_gap, unemp_gap, trend_rate]

# ------------------------------------------------------
# 7) EXTRACT / COMPUTE TERM PREMIUM ESTIMATES
# ------------------------------------------------------
# One simplified approach (not exactly as in the paper) is:
# term premium (TP) for the "long" yield at time t
# = y_long(t) - [expected average short rate over that horizon].
#
# In a more advanced setting, you'd forecast future short yields from the
# model, then average them. Here, we do a naive single-step attempt:
cyclical_level_est = state_means[:, 0]
slope_est = state_means[:, 1]
trend_level_est = state_means[:, 4]

# Reconstruct the short rate from states:
model_short = trend_level_est + cyclical_level_est

# Similarly, reconstruct the model long yield:
model_long = trend_level_est + cyclical_level_est + slope_est

# Approximate "risk-neutral short rate" over 10y (toy approach) as just
# the current short rate estimate (in reality, you'd do forward iteration).
average_short_rate_10y = model_short  # (placeholder)

# Term premium = observed long yield - average of future short yields
term_premium = y_long - average_short_rate_10y

# For demonstration, let's print the last estimate:
print("Last period estimates:")
print(f"Short yield model estimate: {model_short[-1]:.4f}")
print(f"Long yield model estimate: {model_long[-1]:.4f}")
print(f"Observed Long yield: {y_long[-1]:.4f}")
print(f"Term premium estimate: {term_premium[-1]:.4f}")