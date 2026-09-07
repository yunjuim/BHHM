using LinearAlgebra, StatsBase, Distributions, Random, SpecialFunctions

include("MFM.jl")

# Normal distribution
const constant = 0.5 * log(2.0 * pi)
normpdf(x) = exp(-0.5 * x * x - constant)
normpdf(x, mu, sigma) = normpdf((x - mu) / sigma) / sigma
log_normpdf(x, mu, sigma) = (z = (x - mu) / sigma;  -0.5 * z * z - log(sigma) - constant)

mutable struct g
    n::Int64 
    
    sum_y::Float64
    
    sum_yy::Float64
    sum_xx::Array{Float64, 2}
    sum_xy::Array{Float64, 1}
    
    g(p) = (gg = new(); gg.n = 0; gg.sum_y = 0.0; gg.sum_yy = 0.0; gg.sum_xx = zeros(p, p); gg.sum_xy = zeros(p); gg)
end

g_clear!(gg) = (gg.n = 0; gg.sum_y = 0.0; gg.sum_yy = 0.0; gg.sum_xx[:, :] .= 0.; gg.sum_xy[:] .= 0.)
g_add!(gg, y, x) = (gg.n += 1; gg.sum_y += y; gg.sum_yy += y * y; gg.sum_xx += x * x'; gg.sum_xy[:] += x * y)
g_remove!(gg, y, x) = (gg.n -= 1; gg.sum_y -= y; gg.sum_yy -= y * y; gg.sum_xx -= x * x'; gg.sum_xy[:] -= x * y)

b_clear!(beta1) = (beta1[:] .= 0.0)

f_logdet(A::Cholesky) = 2 * sum(log.(diag(A.U)))
f_logdet(A::Cholesky, nw, rw, s2, s2b_) = 2 * (sum(log.(diag(A.U))) + 0.5 * (nw - rw) * log(s2 * s2b_))
f_logdet(A::Cholesky, nw, rw, t, s2, s2b_) = 2 * (sum(log.(diag(A.U))) + 0.5 * (nw - rw) * log(t * s2 * s2b_))

function f_xy(x, y) # x .* y
    z = zeros(Bool, length(x))
    @inbounds for s = 1 : length(x); z[s] = x[s] * y[s]; end; z
end

function f_xy(x, y, c) # const * x .* y
    z = zeros(Bool, length(x))
    @inbounds for s = 1 : length(x); z[s] = c * x[s] * y[s]; end; z
end

function f_add_diagnal(x, v, c)
    @inbounds for s = 1 : sum(v); x[s, s] += c; end
    x
end

function f_xx_rw(x, r, w, wc, c)
    xx = x[w, wc]
    rr = r[w] * r[wc]'

    for j = 1 : size(xx)[2]; 
        for i = 1 : size(xx)[1]; 
            xx[i, j] = c * xx[i, j] * rr[i, j]; 
        end; 
    end
    xx
end

function f_xx_rw(x, r, w, wc)
    xx = x[w, wc]
    rr = r[w] * r[wc]'

    for j = 1 : size(xx)[2]; 
        for i = 1 : size(xx)[1]; 
            xx[i, j] = xx[i, j] * rr[i, j]; 
        end; 
    end
    xx
end

function f_xy_rw(x, r, l, c)
    xx, rr = x[l], r[l]
    for j = 1 : length(xx); 
        xx[j] = c * xx[j] * rr[j]
    end
    return xx
end

function f_xy_rw(x, r, l)
    xx, rr = x[l], r[l]
    for j = 1 : length(xx); 
        xx[j] = xx[j] * rr[j]
    end
    return xx
end

function log_likelihood(y, x, b, Eta, r, w, s2)
    mu = 0.0
    for j = 1 : length(x);
        if r[j] == true
            if w[j] == true
                mu += x[j] * b[j]
            else 
                mu += x[j] * Eta[j]
            end
        end
    end 
    
    log_normpdf(y, mu, sqrt(s2))
end

function log_prior(b1, w, s2b, p)
    xx = 0.0
    for j = 1 : p; if w[j] == true; xx += log_normpdf(b1[j], 0.0, sqrt(s2b)); end; end
    return xx
end

function prior_sample!(b, n_w, w, s2b)
    x = zeros(n_w)
    for i = 1 : n_w; x[i] = rand(Normal(0, sqrt(s2b))); end
    b[w] = x
end

function update_r!(gs, z, r, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_, p, rho_r)

    for jj in 2 : p   
        logO = log_hr(jj, gs, z, r, false, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_) - 
                log_hr(jj, gs, z, r, true, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_)
        logO += (log(1 - rho_r[jj]) - log(rho_r[jj]))
        success_p = 1 / (1 + exp(logO))
        r[jj] = rand(Bernoulli(success_p))
    end
    return r
end

function log_hr(jj, gs, z, r, r0, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_)
    
    r[jj] = r0
    rw, rwc = f_xy(r, w), f_xy(r, wc)
    n_rw, n_rwc = sum(rw), sum(rwc)
    
    sumlogdet_Swd_, sum_xy_Swd_xy = 0.0, 0.0
    E_, m = zeros(n_wc, n_wc), zeros(n_wc)
    
    for j in 1 : t; 
    
        xy_rw = f_xy_rw(gs[j].sum_xy, r, w)
        xy_rwc = f_xy_rw(gs[j].sum_xy, r, wc)
        xx_rw_rwc = f_xx_rw(gs[j].sum_xx, r, w, wc)
        
        # Compute inv(Sigma_d)
        Swd_ = f_xx_rw(gs[j].sum_xx, r, w, w)
        Swd_ = f_add_diagnal(Swd_, w, s2 * s2b_)
        Swd_ = (Swd_ + Swd_') * 0.5
        Swd_chol = cholesky(Swd_)
        sumlogdet_Swd_ += f_logdet(Swd_chol) 

        sum_xy_Swd_xy += dot(xy_rw, Swd_chol \ xy_rw) # Rxw_y' * inv(Swd_) * Rxw_y
        
        # Compute Omega_d
        Od_ = f_xx_rw(gs[j].sum_xx, r, wc, wc)
        c1 = Swd_chol \ xx_rw_rwc
        c1t = Matrix(c1')
        c2 = c1t * xx_rw_rwc
        Od_ -= c2

        E_ += (Od_ + Od_') * 0.5
        m += xy_rwc - c1t * xy_rw

    end
    
    E_ = f_add_diagnal(E_, wc, s2 * s2eta_)
    E_ = (E_ + E_') * 0.5
    E_chol = cholesky(E_)

    - 0.5 * sumlogdet_Swd_ + 0.5 * s2_ * sum_xy_Swd_xy - 0.5 * f_logdet(E_chol) + 0.5 * s2_ * dot(m, E_chol \ m)
end
function update_w!(gs, z, r, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_, p, rho_w)

    for jj in 2 : p

        logO = log_hw(jj, gs, z, r, w, wc, false, t, s2, s2_, s2b_, s2eta_, p) - log_hw(jj, gs, z, r, w, wc, true, t, s2, s2_, s2b_, s2eta_, p) 
        logO += (log(1 - rho_w[jj]) - log(rho_w[jj]))
        success_p = 1 / (1 + exp(logO))
    
        w[jj] = rand(Bernoulli(success_p))
        wc[jj] = !w[jj]
    end
    
    n_w, n_wc = sum(w), sum(wc)
    return w, wc, n_w, n_wc
end

function log_hw(jj, gs, z, r, w, wc, w0, t, s2, s2_, s2b_, s2eta_, p)

    w[jj] = w0    
    wc[jj] = !w0
    
    rw, rwc = f_xy(r, w), f_xy(r, wc)
    n_rw, n_rwc = sum(rw), sum(rwc)
    n_w, n_wc = sum(w), sum(wc)
    
    sumlogdet_Swd_, sum_xy_Swd_xy = 0.0, 0.0
    E_, m_eta = zeros(n_wc, n_wc), zeros(n_wc)

    for j = 1 : t
        
        xy_rw = f_xy_rw(gs[j].sum_xy, r, w)
        xy_rwc = f_xy_rw(gs[j].sum_xy, r, wc)
        xx_rw_rwc = f_xx_rw(gs[j].sum_xx, r, w, wc)
    
        # Compute inv(Sigma_d)
        Swd_ = f_xx_rw(gs[j].sum_xx, r, w, w)
        Swd_ = f_add_diagnal(Swd_, w, s2 * s2b_)
        Swd_ = (Swd_ + Swd_') * 0.5
        Swd_chol = cholesky(Swd_)
        sumlogdet_Swd_ += f_logdet(Swd_chol) 
    
        sum_xy_Swd_xy += dot(xy_rw, Swd_chol \ xy_rw) # Rxw_y' * inv(Swd_) * Rxw_y
    
        # Compute Omega_d
        Od_ = f_xx_rw(gs[j].sum_xx, r, wc, wc)
        c1 = Swd_chol \ xx_rw_rwc
        c1t = Matrix(c1')
        c2 = c1t * xx_rw_rwc
        Od_ -= c2

        E_ += (Od_ + Od_') * 0.5
        m_eta += xy_rwc - c1t * xy_rw
    
    end

    E_ = f_add_diagnal(E_, wc, s2 * s2eta_)
    E_chol = cholesky(E_)

    0.5 * n_w * log(s2b_) + 0.5 * n_wc * log(s2eta_) - 0.5 * sumlogdet_Swd_ + 0.5 * s2_ * sum_xy_Swd_xy - 0.5 * f_logdet(E_chol) + 0.5 * s2_ * dot(m_eta, E_chol \ m_eta)
end

function update_beta(g, z, r, w, wc, n_w, eta1, s2_, s2b_, p)
    
    B = zeros(p)    
    xy_rw = f_xy_rw(g.sum_xy, r, w, s2_)
    xx_rw_rwc = f_xx_rw(g.sum_xx, r, w, wc, s2_)
    
    # Compute inv(Sigma_d)
    Swd_ = f_xx_rw(g.sum_xx, r, w, w, s2_)
    Swd_ = f_add_diagnal(Swd_, w, s2b_)
    Swd_chol = cholesky(Swd_)
    
    xx_rw_rwc_eta = xx_rw_rwc * eta1
    m = Swd_chol \ (xy_rw - xx_rw_rwc_eta)
    
    rn = randn(n_w)
    b = m + Swd_chol.U \ rn
    B[w] = b
    B
end

function update_eta(gs, z, r, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_, p)
    
    Eta = zeros(p)
    rw, rwc = f_xy(r, w), f_xy(r, wc)
    E_, m_eta = zeros(n_wc, n_wc), zeros(n_wc)

    for j = 1 : t
        
        xy_rw = f_xy_rw(gs[j].sum_xy, r, w)
        xy_rwc = f_xy_rw(gs[j].sum_xy, r, wc)
        xx_rw_rwc = f_xx_rw(gs[j].sum_xx, r, w, wc)
    
        # Compute inv(Sigma_d)
        Swd_ = f_xx_rw(gs[j].sum_xx, r, w, w)
        Swd_ = f_add_diagnal(Swd_, w, s2 * s2b_)
        Swd_chol = cholesky(Swd_)
    
        # Compute Omega_d
        Od_ = f_xx_rw(gs[j].sum_xx, r, wc, wc)
        c1 = Swd_chol \ xx_rw_rwc
        c1t = Matrix(c1')
        c2 = c1t * xx_rw_rwc
        Od_ -= c2

        E_ += (Od_ + Od_') * 0.5
        m_eta += xy_rwc - c1t * xy_rw
    
    end
    
    E_ = f_add_diagnal(E_, wc, s2 * s2eta_)
    E_chol = cholesky(E_)
    m = E_chol \ m_eta
    
    rn = randn(n_wc)
    eta1 = m + E_chol.U \ rn
    Eta[wc] = eta1
    
    return Eta, eta1
end

function update_s2(y, x, z, bs, Eta, r, w, n, p, a0, b0)
    
    sum_y_mu2 = 0.0 
    for i in 1 : n; 
        mu = 0.0
        cc = z[i]
        bb = bs[cc]
        xx = x[i]
        for j = 1 : p;
            if r[j] == true
                if w[j] == true
                    mu += xx[j] * bb[j]
                else 
                    mu += xx[j] * Eta[j]
                end
            end
        end 
        
        y_mu = y[i] - mu
        sum_y_mu2 += y_mu * y_mu
    end
    
    s2 = rand(InverseGamma(0.5 * n + a0, 0.5 * sum_y_mu2 + b0))    
    s2
end

mutable struct hp
    n::Int32
    p::Int32
    
    ar::Float64
    br::Float64
    aw::Float64
    bw::Float64
    
    s2b::Float64
    s2eta::Float64
    
    s2b_::Float64
    s2eta_::Float64
    
    a0::Float64 # s2 ~ InvGamma(a0, b0)
    b0::Float64
    
    t_max::Int32
end

function construct_hp(x)
    n = size(x)[1]
    p = length(x[1])
    
    ar, br, aw, bw = 1.0, 1.0, 1.0, 1.0
    s2b, s2eta = 10.0, 10.0
    s2b_, s2eta_ = 1 / 10, 1 / 10
    a0, b0 = 1.0, 0.0001 # sigma2 ~ IG(a0, b0)
    t_max = 20
    
    return hp(n, p, ar, br, aw, bw, s2b, s2eta, s2b_, s2eta_, a0, b0, t_max)
end


lgamma_(x) = logabsgamma(x)[1]

function run_sampler(y, x, H;
        
        # MFM options: 
        gamma = 1.0, # Dirichlet_k(gamma,...,gamma)
        log_pk = "k -> log(0.5) + (k-1) * log(0.5)", # string representation of log(p(k))
        
        use_splitmerge = false, n_split = 5,  n_merge = 5,
        
        n_total = 5000, t_max = 20, 
        Initial_z = Initial_z
    )

    n, p = H.n, H.p
    ar, br, aw, bw = H.ar, H.br, H.aw, H.bw
    s2b, s2eta = H.s2b, H.s2eta
    s2b_, s2eta_ = H.s2b_, H.s2eta_
    a0, b0 = H.a0, H.b0

    # Initial values for rho and indicator variables
    w, r = zeros(Bool, p), zeros(Bool, p)
    rho_r, rho_w = rand(Beta(ar, br), p), rand(Beta(aw, bw), p)
    Ip = Diagonal(ones(p))
    
    r[1], w[1] = 1, 1 # r and w for an itcp
    for i in 2 : p; 
        r[i] = false; 
        w[i] = false; 
    end

    wc = .!w 
    n_w, n_wc = sum(w), sum(wc)
    
    bs = [zeros(p) for i = 1 : t_max]
    Eta = zeros(p)
    for i in 2 : p; if r[i] * wc[i] != 0; Eta[i] = rand(Normal(0, sqrt(s2eta))); end; end
    eta1 = Eta[wc]
    
    # Initial values for clustering
    list = zeros(Int, t_max + 3); 
    #z = ones(Int, n); t = 1; list[1] = 1; c_next = 2
    #list[1:2] = 1:2; t = 2; z = copy(true_z); c_next = 3  # an available cluster ID to be used next
    t = length(unique(Initial_z))
    list[1:t] = 1:t; z = copy(Initial_z); c_next = t + 1  # an available cluster ID to be used next
    
    N = zeros(Int, t_max); 
    gs = [g(p) for i = 1 : t_max]
    for j = 1 : t; for i = 1 : n; if z[i] == j; N[j] += 1; g_add!(gs[j], y[i], x[i]); end; end; end

    lpk = eval(Meta.parse(log_pk))
    log_pk_fn(k) = Base.invokelatest(lpk,k)
    log_v = MFM.coefficients(log_pk_fn,gamma,n,t_max+1)
    a = b = gamma
    log_p = zeros(n+1)
    log_Nb = log.((1:n) .+ b);
    
    s2 = rand(InverseGamma(a0, b0))
    s2_ = 1 / s2
    
    z_r, N_r, r_r, w_r = zeros(Int64, n, n_total), zeros(Int64, t_max, n_total), zeros(Bool, p, n_total), zeros(Bool, p, n_total)
    eta_r, b_r = spzeros(p, n_total), [spzeros(p, n_total) for i = 1 : t_max] #zeros(p, n_total), zeros(p, n_total, t_max)
    t_r, s2_r = zeros(Int32, n_total), zeros(n_total);
    rho_r_r, rho_w_r = zeros(p, n_total), zeros(p, n_total)
    
    println("n = $n, n_total = $n_total")
    print("Running... ") 
    
    elapsed_time = (
    @elapsed for iter = 1 : n_total
            
        update_r!(gs, z, r, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_, p, rho_r)
        w, wc, n_w, n_wc = update_w!(gs, z, r, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_, p, rho_w)    
        Eta, eta1 = update_eta(gs, z, r, w, wc, t, n_w, n_wc, s2, s2_, s2b_, s2eta_, p)
        for i = 1 : t;  ii = list[i]; bs[ii] = update_beta(gs[ii], z, r, w, wc, n_w, eta1, s2_, s2b_, p); end

        for i = 1 : n
            # remove point i from it's cluster
            c = z[i]        
            N[c] -= 1
            g_remove!(gs[c], y[i], x[i])

            if N[c] > 0
                c_prop = c_next
                prior_sample!(bs[c_prop], n_w, w, H.s2b)
            else
                c_prop = c
                # remove cluster {i}, keeping list in proper order
                ordered_remove!(c, list, t)         
                t -= 1
            end

            # compute probabilities for resampling
            for j = 1:t; cc = list[j]
                log_p[j] = log_Nb[N[cc]] + log_likelihood(y[i], x[i], bs[cc], Eta, r, w, s2)
            end
            log_p[t+1] = log_v[t+1]-log_v[t] + log(a) + log_likelihood(y[i], x[i], bs[c_prop], Eta, r, w, s2)
            j = randlogp!(log_p, t + 1)

            # add point i to it's new cluster
            if j<=t           
                c = list[j]
                b_clear!(bs[c_prop])
            else
                c = c_prop
                ordered_insert!(c,list,t)
                t += 1
                c_next = ordered_next(list)
                @assert(t <= t_max, "Sampled t has exceeded t_max. Increase t_max and retry.")
            end

            g_add!(gs[c], y[i], x[i])
            z[i] = c
            N[c] += 1
        end           
            
        # update s2            
        s2 = update_s2(y, x, z, bs, Eta, r, w, n, p, a0, b0)
        s2_ = 1 / s2
            
        # update rho_r and rho_w
        for jj in 2 : p; 
            rho_r[jj] = rand(Beta(r[jj] + ar, 1 - r[jj] + br));
            rho_w[jj] = rand(Beta(w[jj] + aw, 1 - w[jj] + bw)); 
        end
            
        N_r[:, iter], z_r[:, iter], r_r[:, iter], w_r[:, iter] = N, z, r, w
        eta_r[:, iter] = Eta
        for i = 1 : t; ii = list[i]; b_r[ii][:, iter] = bs[ii]; end #b_r[:, iter, ii] = bs[ii]; end
        rho_r_r[:, iter], rho_w_r[:, iter] = rho_r, rho_w
        s2_r[iter] = s2
        t_r[iter] = t
    end
    )
    println("complete.")
    println("Elapsed time = $elapsed_time seconds")
    
    return Result(t_r, z_r, N_r, r_r, w_r, b_r, eta_r, rho_r_r, rho_w_r, s2_r, elapsed_time)
end

struct Result
    t::Array{Int32, 1}
    z::Array{Int64, 2}
    N::Array{Int64, 2}
    r::Array{Bool, 2}
    w::Array{Bool, 2}
    b::Vector{SparseMatrixCSC{Float64, Int64}}
    eta::SparseMatrixCSC{Float64, Int64}
    rho_r::Array{Float64, 2}
    rho_w::Array{Float64, 2}
    s2::Array{Float64, 1}
    elapsed_time::Float64
end