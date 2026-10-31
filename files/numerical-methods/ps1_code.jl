
# student: Alexandre V. P Nascimento  
# ------------------------------------------ #
# Problem Set 1 - Numerical Methods FGV EPGE #
# ------------------------------------------ #

# Main Packages
#Pkg.add("UnPack")
using Pkg
using Random
using Distributions
using LinearAlgebra
using Statistics
using Plots
using DataFrames
using GLM
using Parameters # allows @unpack
using UnPack


# ------------------------------------------ #
# Defining parameters                        #
# ------------------------------------------ #

function setPar(;
    ρ = 0.95,    # persistence
    σ = 0.007,   # standard deviation
    Nz_grid = 9, # number of grid points
    m = 3.0,     # scaling parameter
    T = 10000,   # simulation periods 
    #z0 = 0.0,   # AR(1) process initial value
    seed = 42   # random seed
    )
    #return (ρ = ρ, σ = σ, N = N, m = m, T = T, z0 = z0, seed = seed)
    return (ρ = ρ, σ = σ, Nz_grid = Nz_grid, m = m, T = T, seed = seed)
end

par = setPar()


"------------------------------------------------------------------------------------------------
1. Discretize the process using the Tauchen (1986) method. Use 9 points.
------------------------------------------------------------------------------------------------"

# -------------------------------------- #
# Tauchen's Method                       #
# -------------------------------------- #

function tauchen_function(; parameters)
    @unpack ρ, σ, m, Nz_grid = parameters

    # Bounds 
    z_upper = m * (σ / sqrt(1 - ρ^2))  
    z_lower = -z_upper   

    # Grid
    z_grid = collect(LinRange(z_lower, z_upper, Nz_grid)) # collect turns z_grid into a Vector
    Δz = z_grid[2] - z_grid[1]

    # Empity transition matrix 
    P = zeros(Nz_grid, Nz_grid)

    for i in 1:Nz_grid
        for j in 1:Nz_grid

            if j == 1 # lowest grid point
                A = (z_grid[1] + Δz*0.5 - ρ * z_grid[i]) / σ
                P[i, j] = cdf(Normal(), A) # Normal() = N(0,1)

            elseif j == Nz_grid # highest grid point
                B = (z_grid[Nz_grid] - Δz*0.5 - ρ * z_grid[i]) / σ
                P[i, j] = ccdf(Normal(), B) #1 - cdf(Normal(), B)

            else # interior grid points
                C = (z_grid[j] + Δz*0.5 - ρ * z_grid[i]) / σ
                D = (z_grid[j] - Δz*0.5 - ρ * z_grid[i]) / σ

                P[i, j] = cdf(Normal(), C) - cdf(Normal(), D)
            end
        end
    end

    return z_grid, P
end

# Tests: 
# Construct the grid and transition matrix.
z_tauchen, P_tauchen = tauchen_function(parameters = par)

# Dimensions:
@assert length(z_tauchen) == par.Nz_grid
@assert size(P_tauchen) == (par.Nz_grid, par.Nz_grid)

# Check if each row sums to one:
@assert all(0 .<= P_tauchen .<= 1)
@assert maximum(abs.(sum(P_tauchen, dims = 2) .- 1)) < 1e-12

# Main Results:
z_tauchen'
P_tauchen




"------------------------------------------------------------------------------------------------
2. Discretize the process using the Rouwenhorst method. Use 9 points.
------------------------------------------------------------------------------------------------"
# -------------------- #
# Rouwenhorst's Method #
# -------------------- #
function rouwenhorst_function(; parameters)
    @unpack ρ, σ, Nz_grid = parameters
    N = Nz_grid

    # Step 1: Bonds 
    σ_z = σ/sqrt(1-ρ^2)
    z_upper = σ_z * sqrt(N-1)
    z_lower = -z_upper 
    z_grid = collect(LinRange(z_lower, z_upper, N))

    #Step 2: Compute P recursively 
    p = (1+ρ)/2 
    P_old = [p 1-p; 1-p p] #2x2
    for n in 3:N 
        #P_new = zeros(n, n)
        A, B, C, D = zeros(n, n), zeros(n, n), zeros(n, n), zeros(n, n)
        A[1:n-1, 1:n-1] = P_old 
        B[1:n-1, 2:n] = P_old 
        C[2:n, 1:n-1] = P_old 
        D[2:n, 2:n] = P_old 
        P_new = p*(A + D) + (1-p)*(B + C)

    # Step 3: Normalization in each interaction 
        P_new[2:n-1, :] =  P_new[2:n-1, :] ./ 2
        P_old = P_new
    end

    # Final rouwenhorst Matrix
    P = P_old 
    return z_grid, P
end

z_rouwenhorst, P_rouwenhorst = rouwenhorst_function(parameters = par)

# Checks:
@assert length(z_rouwenhorst) == par.Nz_grid
@assert size(P_rouwenhorst) == (par.Nz_grid, par.Nz_grid)
@assert all(0 .<= P_rouwenhorst .<= 1)
@assert maximum(abs.(sum(P_rouwenhorst, dims = 2) .- 1)) < 1e-12

# Main results: 
z_rouwenhorst'
P_rouwenhorst


"------------------------------------------------------------------------------------------------
3. Simulate the continuous process for 10000 periods. Do the same for the discretized processes 
(remember to use the same realizations for the shock for all simulations). Compare the paths of 
each process (graphs are useful here). If they’re not very close, use more points.
-------------------------------------------------------------------------------------------------"

# --------------------------------------------- #
# Generating the shocks for the AR(1) process   #
# --------------------------------------------- #
Random.seed!(par.seed)
ε = rand(Normal(0, par.σ), par.T)

# ------------------ #
# AR(1) process      #
# ------------------ #

function AR1_process(; ε, parameters)
    @unpack ρ, T = parameters

    z = zeros(T)
    for t in 1:(T-1)
        z[t+1] = ρ*z[t] + ε[t+1]
    end
    return z
end

# Test.
z_AR1 = AR1_process(; ε = ε, parameters = par)
plot(1:par.T, z_AR1, label = "AR(1)", xlabel = "T")


# ------------------ #
# AR(1) discretized  #
# ------------------ #

function AR1_discretized(method; parameters, uniforms)
    @unpack T, Nz_grid = parameters
    N = Nz_grid

    z_sim = zeros(T)
    state_index = zeros(Int, T)

    # Choose discretization method.
    if method == :Tauchen
        z_grid, P_matrix = tauchen_function(parameters = parameters)

    elseif method == :Rouwenhorst
        z_grid, P_matrix = rouwenhorst_function(parameters = parameters)

    else
        error("Method not found. Choose :Tauchen or :Rouwenhorst.")
    end

    # Initial state: middle grid point (zero for our nine-state grid).
    state_index[1] = div(N + 1, 2)
    z_sim[1] = z_grid[state_index[1]]

    for t in 2:T
        current_state = state_index[t - 1]
        prob = P_matrix[current_state, :]
        cumulative_probs = cumsum(prob)
        cumulative_probs[end] = 1.0 # Avoid rounding errors at the endpoint.

        u = uniforms[t - 1]
        next_state = searchsortedfirst(cumulative_probs, u)

        state_index[t] = next_state
        z_sim[t] = z_grid[next_state]
    end

    return z_sim
end

# Shared shocks.
par = setPar()
Random.seed!(par.seed)
ε = rand(Normal(0, par.σ), par.T)

# Transform the normal shocks used by the AR(1) into uniform draws.
uniforms = cdf.(Normal(0, par.σ), ε[2:end])

# Simulations:
z_AR1 = AR1_process(; parameters = par, ε = ε)
z_tauchen_sim = AR1_discretized(:Tauchen; parameters = par, uniforms = uniforms)
z_rouwenhorst_sim = AR1_discretized(:Rouwenhorst; parameters = par, uniforms = uniforms)

# Plots:
plot(1:par.T, z_tauchen_sim, label = "Tauchen")
plot!(1:par.T, z_AR1, label = "AR(1)", xlabel = "T", ylabel = "z", title = "ρ = $(par.ρ)")

plot(1:par.T, z_rouwenhorst_sim, label = "Rouwenhorst")
plot!(1:par.T, z_AR1, label = "AR(1)", xlabel = "T", ylabel = "z", title = "ρ = $(par.ρ)")


"------------------------------------------------------------------------------------------------
4. Estimate the AR(1) processes based on simulated data, both for Tauchen and Rouwenhorst. 
How close are they from the real data generating process? If they’re not very close, use 
more points.
------------------------------------------------------------------------------------------------"

#=
Let's run a regression of z_t on z_{t-1} to estimate ρ_hat for each simulated process.
The original process is:
    z_t = ρ * z_{t-1} + ε_t
Therefore, we estimate the following regression:
    z_t = α + ρ_hat * z_{t-1} + u_t
If the discretization is accurate, ρ_hat should be close to the true value of ρ.
=#

function estimate_rho(z)
    data = DataFrame(z_t = z[2:end], z_lag = z[1:end-1])
    model = lm(@formula(z_t ~ z_lag), data)
    result = coef(model)[2]
    return result
end

rho_AR1 = estimate_rho(z_AR1)
rho_tauchen = estimate_rho(z_tauchen_sim)
rho_rouwenhorst = estimate_rho(z_rouwenhorst_sim)


qt_digits = 6
results = DataFrame(
    ρ      = [par.ρ, "",""],
    Method = ["AR(1)", "Tauchen", "Rouwenhorst"],
    μ = round.([mean(z_AR1), mean(z_tauchen_sim), mean(z_rouwenhorst_sim)], digits = qt_digits),
    σ = round.([std(z_AR1), std(z_tauchen_sim), std(z_rouwenhorst_sim)], digits = qt_digits),
    ρ_hat = round.([rho_AR1, rho_tauchen, rho_rouwenhorst], digits = qt_digits),
    ρ_error = round.([abs(rho_AR1 - par.ρ), abs(rho_tauchen - par.ρ), abs(rho_rouwenhorst - par.ρ)], digits = qt_digits)
)



"------------------------------------------------------------------------------------------------
5. Redo the questions above with ρ = 0.99.
------------------------------------------------------------------------------------------------"

# Setting the new ρ = 0.99
par_new = setPar(ρ = 0.99, T = par.T)

# 5.1–5.2. Discretize the process with nine states.
z_tauchen_99, P_tauchen_99 = tauchen_function(parameters = par_new)
z_rouwenhorst_99, P_rouwenhorst_99 = rouwenhorst_function(parameters = par_new)

@assert maximum(abs.(sum(P_tauchen_99, dims = 2) .- 1)) < 1e-12
@assert maximum(abs.(sum(P_rouwenhorst_99, dims = 2) .- 1)) < 1e-12

# ----------------- #
# Tauchen, ρ = 0.99 #
# ----------------- #
z_tauchen_99'
P_tauchen_99

# --------------------- #
# Rouwenhorst, ρ = 0.99 #
# --------------------- #
z_rouwenhorst_99'
P_rouwenhorst_99

# 5.3. Simulate 10,000 periods using the same draws as before.
z_AR1_99 = AR1_process(; ε = ε, parameters = par_new)
z_tauchen_sim_99 = AR1_discretized(:Tauchen; parameters = par_new, uniforms = uniforms)
z_rouwenhorst_sim_99 = AR1_discretized(:Rouwenhorst; parameters = par_new, uniforms = uniforms)

# Plots:
plot(1:par_new.T, z_tauchen_sim_99, label = "Tauchen")
plot!(1:par_new.T, z_AR1_99, label = "AR(1)", xlabel = "T", ylabel = "z", title = "ρ = $(par_new.ρ)")

plot(1:par_new.T, z_rouwenhorst_sim_99, label = "Rouwenhorst")
plot!(1:par_new.T, z_AR1_99, label = "AR(1)", xlabel = "T", ylabel = "z", title = "ρ = $(par_new.ρ)")




# 5.4. Estimate persistence and compare sample moments.
rho_AR1_99 = estimate_rho(z_AR1_99)
rho_tauchen_99 = estimate_rho(z_tauchen_sim_99)
rho_rouwenhorst_99 = estimate_rho(z_rouwenhorst_sim_99)

qt_digits = 6
results_99 = DataFrame(
    ρ = fill(par_new.ρ, 3),
    Method = ["AR(1)", "Tauchen", "Rouwenhorst"],
    μ = round.([mean(z_AR1_99), mean(z_tauchen_sim_99), mean(z_rouwenhorst_sim_99)], digits = qt_digits),
    std_z = round.([std(z_AR1_99), std(z_tauchen_sim_99), std(z_rouwenhorst_sim_99)], digits = qt_digits),
    ρ_hat = round.([rho_AR1_99, rho_tauchen_99, rho_rouwenhorst_99], digits = qt_digits),
    ρ_error = round.([abs(rho_AR1_99 - par_new.ρ), abs(rho_tauchen_99 - par_new.ρ), abs(rho_rouwenhorst_99 - par_new.ρ)], digits = qt_digits)
)
results_99
s