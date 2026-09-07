# BHHM

## Publication

Yunju Im and Aixin Tan. (2026). Bayesian Mixture Models with Structured Sparsity for Cancer Heterogeneity Analysis. Manuscript

## Description

1. `MFM.jl`, `ftns.jl`, and `ftns-htr-hm-MFM.jl` contain the main and supporting functions for generating MCMC samples from the proposed heterogeneous-homogeneous mixture-of-finite-mixtures model.
2. `dat1.csv` contains the simulated dataset used in the toy example.

## Examples

Below is an example illustrating how to run the proposed model.

```julia
using StatsBase, Statistics, LinearAlgebra, Distributions, Random
using CSV, DataFrames, JLD2

# Load model functions and simulated data
include("MFM.jl")
include("ftns.jl")
include("ftns-htr-hm-MFM.jl")

simdat = CSV.read("dat1.csv", DataFrame)

# Extract the outcome, true subgroup labels, and predictors
y = simdat[:, "y"]
true_z = Int.(simdat[:, "z"])
X = Matrix(select(simdat, r"^x"))

n = length(y)
x = [X[i, :] for i in 1:n]

# Construct hyperparameters
function construct_hp(x)
    n = length(x)
    p = length(x[1])

    ar, br, aw, bw = 1.0, 1.0, 1.0, 1.0
    s2b, s2eta = 10.0, 10.0
    s2b_, s2eta_ = 1 / 10, 1 / 10
    a0, b0 = 1.0, 0.0001
    t_max = 20

    return hp(
        n, p, ar, br, aw, bw,
        s2b, s2eta, s2b_, s2eta_,
        a0, b0, t_max,
    )
end

H = construct_hp(x)

# MCMC settings
n_total = 10000
n_burn = 5000

Random.seed!(1)
z0 = ones(Int64, n)

# Run posterior sampler
s1 = run_sampler(
    y,
    x,
    H,
    n_total=n_total,
    Initial_z=z0,
    t_max=26,
    log_pk="k -> log(0.5) + (k-1) * log(0.5)",
)

# Posterior inclusion probabilities conditional on two subgroups
id = findall(==(2), s1.t[(n_burn + 1):n_total]) .+ n_burn
htr = s1.w .* s1.r
hm = (.!s1.w) .* s1.r

pip = vec(mean(s1.r[2:end, id], dims=2))
pip_htr = vec(mean(htr[2:end, id], dims=2))
pip_hm = vec(mean(hm[2:end, id], dims=2))

# Selected heterogeneous and homogeneous predictors
selected_htr = findall(>(0.5), pip_htr)
selected_hm = findall(>(0.5), pip_hm)
```

For more information, please contact Yunju Im at yim@unmc.edu.
