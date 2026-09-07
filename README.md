# BHHM

## Publication

Yunju Im and Aixin Tan. (2026). Bayesian Mixture Models with Structured Sparsity for Cancer Heterogeneity Analysis. Manuscript

## Description

1. `MFM.jl`, `ftns.jl`, and `ftns-BHHM.jl` contain the main and supporting functions for implementing the proposed BHHM method and generating posterior samples.
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
H = construct_hp(x)

# MCMC settings
n_total, n_burn = 2000, 500


# Run posterior sampler
s1 = run_sampler(y, x, H, n_total = n_total, Initial_z = ones(Int64, n), t_max=26, log_pk="k -> log(0.5) + (k-1) * log(0.5)")

# Posterior inclusion probabilities conditional on two subgroups
id = findall(==(2), s1.t[(n_burn + 1):n_total]) .+ n_burn
r_htr = s1.w .* s1.r
r_hm = (.!s1.w) .* s1.r

# Posterior inclusion probabilities
pip = vec(mean(s1.r[2:end, id], dims=2))
```

For more information, please contact Yunju Im at yim@unmc.edu.
