# This code solves the one-period Eaton-Gersovitz model with 
# one-period debt allowing for Epstein-Zin recursive utility. 

using Parameters
using SpecialFunctions
using Distances
using LinearAlgebra
using SparseArrays

const _TOL = 10.0^(-12)
const _MAX_ITERS = 10000 


###############################################################################
#
# Structs and constructors
# 
#


# `EatonGersovitz` contains the basic parameters of the model as well 
# as some other values and the debt grid. 
@with_kw struct EatonGersovitzModel{M, F1, F2, F3} @deftype Float64
    R = 1.017
    β = 0.953
    γ = 2.0 # risk aversion parameter
    α = 2.0 # 1/IES parameter -- γ == α is CRRA
    θ = 0.282   # prob of regaining market access
    b_max = 1.2   # maximum debt level
    b_min = -0.2   # minimum debt level 
    nb_approx::Int64 = 200  # approximate points for B grid 
    y_process::M = logAR1(
        N=200, 
        ρ=0.948503, 
        σ=0.027093, 
        μ=0.0, 
        span=3.0, 
        inflate_ends=false
    ) 
    # y_process should be a named tuple with a transition matrix, T, a grid, 
    # grid, as well as a value for the ergodic mean, mean. 
    y_def_fun::F1 = arellano_default_cost(0.969, y_process.mean)

    # generated parameters
    ny::Int64 = length(y_process.grid)  # grid points for output 
    y_grid::Array{Float64, 1} = y_process.grid
    T::Array{Float64, 2} = y_process.T
    y_mean = y_process.mean
    y_def_grid::Array{Float64, 1} =  y_def_fun.(y_grid)
    b_grid::Array{Float64, 1} = generate_b_grid(b_min, b_max, nb_approx)
    nb::Int64 = length(b_grid)
    zero_index::Int64 = findfirst(isequal(0.0), b_grid)
    d_and_c_fun::F2 = nothing
    u::F3 = (c -> c^(1 - γ))
    min_v = 0.0
end


struct Allocation{P}
    vD::Array{Float64, 1}
    vR::Array{Float64, 2}
    vMax::Array{Float64, 2}
    q::Array{Float64, 2}
    b_pol::Array{Int64, 2}
    repay::BitArray{2}
    model::P
end 


Allocation(model::EatonGersovitzModel) = Allocation(
    fill(1.0, model.ny),
    fill(1.0, (model.ny, model.nb)),
    fill(1.0, (model.ny, model.nb)),
    fill(1.0, (model.ny, model.nb)),
    zeros(Int64, model.ny, model.nb),
    falses(model.ny, model.nb),
    model 
)


function Base.show(io::IO, model::EatonGersovitzModel)
    @unpack R, β, γ, α, θ, b_max, b_min, nb, ny = model
    @unpack N, ρ, σ, μ, span, inflate_ends = model.y_process.pars
    print(
        io, 
        "R=", R, " β=", β, " γ=", γ, " α=", α," θ=", θ,
        " b_max=", b_max, " b_min=", b_min, " nb=", nb, 
        " N=", N, " ρ=", ρ, " σ=", σ, " μ=", μ, " span=", span, 
        " inflate_ends=", inflate_ends, " ny=", ny
    )    
end


function Base.show(io::IO, alloc::Allocation)
    print(io, "Allocation for model: ")
    show(io, alloc.model)   
end


###############################################################################
#
# Functions related to the constructor of EatonGersovitzModel
#


# from here: https://stackoverflow.com/questions/25678112/insert-item-into-a-sorted-list-with-julia-with-and-without-duplicates
insert_and_dedup!(v::Vector, x) = (splice!(v, searchsorted(v,x), x); v)


function generate_b_grid(b_min, b_max, nB_approx)
    grid = collect(range(b_min, stop = b_max, length = nB_approx))
    insert_and_dedup!(grid, 0.0) # make sure that zero is in the grid of assets
                                 # important for re-entry
    return grid
end 


function arellano_default_cost(par, meanx)
    return x -> min(x, par * meanx)
end 


function proportional_default_cost(par)
    return x -> par * x
end 


function quadratic_default_cost(h1, h2)
    return x -> (x - max(zero(h1), h1 * x + h2 * x^2))
end 


"""
    logAR1(N=100, ρ=0.948503, σ=0.027093, μ=0.0, span=3.0, inflate_ends=true)

    Return a NamedTuple containing the Tauchen discretization of an AR1.
"""
function  logAR1(;
    N = 100, ρ = 0.948503, σ = 0.027093, μ = 0.0, span = 3.0, inflate_ends = true
)
    # Get discretized space using Tauchen's method. Taken from Stelios.

    # Define versions of the standard normal cdf for use in the tauchen method
    std_norm_cdf = x -> 0.5 * erfc(-x/sqrt(2))

    a_bar = span * σ / sqrt(1 - ρ^2)
    y = LinRange(-a_bar, a_bar, N)  # this refers to the log of y
    d = y[2] - y[1]
    # Get transition probabilities
    T = zeros(Float64, N, N)
    if inflate_ends
        for i = 1:N
            # Do end points first
            T[1,i] = std_norm_cdf((y[1] - ρ*y[i] + d/2) / σ)
            T[N,i] = 1 - std_norm_cdf((y[N] - ρ*y[i] - d/2) / σ)
            # fill in the middle columns
            for j = 2:N-1
                T[j, i] = (std_norm_cdf((y[j] - ρ*y[i] + d/2) / σ) -
                    std_norm_cdf((y[j] - ρ*y[i] - d/2) / σ))
            end
        end
    else
        for i = 1:N, j = 1:N
            T[j, i] = (std_norm_cdf((y[j] - ρ*y[i] + d/2) / σ) -
                std_norm_cdf((y[j] - ρ*y[i] - d/2) / σ))
        end
    end
    y_grid = exp.(y .+ μ / (1 - ρ))
    y_mean = exp(μ / (1 - ρ) + 1/2 * σ^2 / (1 - ρ^2))
    T_sums = sum(T, dims=1)
    for i in 1:N
        T[:,i] .*= T_sums[i]^(-1)
    end
    return (
        T=T, grid=y_grid, mean=y_mean, 
        pars=(N=N, ρ=ρ, σ=σ, μ=μ, span=span, inflate_ends=inflate_ends)
    )
end


###############################################################################
#
# Helper functions 
#


function EZutility(model, c, V)
    @unpack β, u, γ, α = model 
    tmp = (1-β) * u(c) + β * V^((1-γ)/(1-α))
    return tmp^(1/(1-γ))
end


function assign!(new::Allocation, iy, ib, vR, vMax, repay, b_pol)
    new.vR[iy, ib] = vR
    new.vMax[iy, ib] = vMax
    new.repay[iy, ib] = repay
    new.b_pol[iy, ib] = b_pol
end


# Helper distance function 
function distance(new::Allocation, old::Allocation)
    error = 0.0
    for (a, b, c, d) in zip(new.vR, old.vR, new.q, old.q)
        error = max(error, abs(a - b), abs(c - d))
    end 
    return error
end


function print_and_stop(dist, i, tol, print_every, max_iters)
    stop  = false
    if mod(i, print_every) == 1
        println("Iteration:", i, " error:", dist)
    end
    if (dist < tol)
        println("Converged!! Iteration:", i, " error: ", dist)
        stop = true 
    end 
    if i > max_iters
        println("DID NOT CONVERGE!! Iteration: ", max_iters, " error: ", dist)
        stop = true
    end
    return stop 
end


###############################################################################
#
# Solver methods
#


function update_vD!(new, model, old; tmp = similar(new.vD))
    @unpack T, θ, zero_index, y_def_grid, y_grid, α = model
    
    #=
    We use vMax to compute vD as it allows the possibility of  
    immediate default after re-entry. 
    Important for iterations away from fixed point. 
    =#
    mm = @view new.vMax[:, zero_index] 
    tmp .= θ .* (mm.^(1-α)) .+ (1-θ) .* (old.vD.^(1-α))
    Threads.@threads for iy in eachindex(y_grid)
        cont_value = 0.0 
         @inbounds for iy′ in eachindex(y_grid)
            cont_value += T[iy′, iy] * tmp[iy′]
         end 
         @inbounds new.vD[iy] = EZutility(model, y_def_grid[iy], cont_value)
    end
end


function update_q!(new, model)
    @unpack T, R = model
    mul!(new.q, T', new.repay)
    new.q .*= R^(-1)
end



function divide_and_conquer!(new, tmp_EV, model, old, iy, (bl, bh))
    if bl + 1 < bh
        bmid = (bl + bh) ÷ 2
        optimize!(new, tmp_EV, model, old, iy, bmid, (new.b_pol[iy, bl], new.b_pol[iy, bh]))

        divide_and_conquer!(new, tmp_EV, model, old, iy, (bl, bmid))
        divide_and_conquer!(new, tmp_EV, model, old, iy, (bmid, bh))
    end
end 


function solve_for_single_iy!(new, tmp_EV, model, old, iy)
    bh = length(model.b_grid)
    optimize!(new, tmp_EV, model, old, iy, 1, (1, bh))
    optimize!(new, tmp_EV, model, old, iy, bh, (new.b_pol[iy, 1], bh))
    
    divide_and_conquer!(new, tmp_EV, model, old, iy, (1, bh))
end 


function optimize!(new, tmp_EV, model, old, iy, ib, (bl, bh))
    @unpack u, β, γ, α, y_grid, b_grid, d_and_c_fun, T, nb, ny, min_v = model

    first_valid = true 
    current_max = min_v
    policy = nb 
    @inbounds for ib_prime in bl:bh
        c = y_grid[iy] + old.q[iy, ib_prime] * b_grid[ib_prime] - 
            b_grid[ib]
        if c > 0.0
            m = ((1 - β) * u(c) + tmp_EV[iy, ib_prime])^(1/(1-γ))
            if (first_valid) || (m > current_max)
                current_max = m 
                policy = ib_prime
                first_valid = false
            end
        end
    end
    if (!first_valid) && (current_max >= old.vD[iy])
        # Found an optimal policy.
        assign!(new, iy, ib, current_max, current_max, true, policy)
    else 
        # No feasible consumption at this debt or default is preferred
        # Assign the default value vD to vMax
        @inbounds assign!(new, iy, ib, current_max, old.vD[iy], false, policy)
    end 
end


function iterate_once!(
    new, model, old; 
    tmp_EV = similar(new.vR), tmp_vMax = similar(new.vR), tmp_EvD = similar(new.vD)
)
    @unpack T, β, γ, α = model

    # auxiliary calculation of expected continuation value function
    tmp_vMax .= old.vMax.^(1-α)  
    mul!(tmp_EV, T', tmp_vMax)
    tmp_EV .= β .* tmp_EV .^ ((1-γ)/(1-α))

    Threads.@threads for iy in 1:model.ny
        solve_for_single_iy!(new, tmp_EV, model, old, iy)
    end 
    update_vD!(new, model, old; tmp=tmp_EvD)
    update_q!(new, model)
end


"""
    solve(model[, new, start, max_iters, tol])

Solve the Eaton-Gersovitz Model
"""
function solve(
    model; 
    new=Allocation(model), 
    start=Allocation(model),
    max_iters::Int64=_MAX_ITERS,
    tol::Float64=_TOL
)
    tmp1, tmp2, tmp3 = similar(new.vR), similar(new.vR), similar(new.vD)
    old = start
    i = 1
    while true 
        iterate_once!(
            new, model, old; 
            tmp_EV = tmp1, tmp_vMax = tmp2, tmp_EvD = tmp3
        )
        # dist = chebyshev(new.vR, old.vR) + chebyshev(new.q, old.q)
        print_and_stop(distance(new, old), i, tol, 100, max_iters) && break 
        new, old = old, new
        i += 1
    end
    return new
end


###############################################################################
#
# Ergodic distrbution methods
#


"""
    create_transition_matrix(alloc)

Construct a sparse matrix containing the transition matrix associated with
alloc. The columns are the current state, the rows are next period state. The
state vector is a 1-D array where position ny * (ib - 1) + iy  corresponds to
(iy, ib), where ny is the total number of y points. That is, we cycle through
the y_grid, and then through the b_grid. This 1-D state vector contains an
additional ny states at its end, representing the default state with each of its
endowment realizations. 
""" 
function create_transition_matrix(alloc)
    @unpack model = alloc 
    @unpack θ, T, y_grid, b_grid, ny, nb = model 

    non_zero_elements = ny * nb + ny * 2
    col_list = Int64[]
    row_list = Int64[]
    val_list = Float64[]

    for iy in eachindex(y_grid)
        for ib in eachindex(b_grid)
            b_pol = alloc.b_pol[iy, ib]
            for iyprime in eachindex(y_grid)
                push!(row_list, ny * (ib - 1) + iy)
                push!(val_list, T[iyprime, iy])
                if alloc.repay[iy, ib] && alloc.repay[iyprime, b_pol]
                    push!(col_list, ny * (b_pol - 1) + iyprime)
                else 
                    push!(col_list, ny * nb + iyprime)  # default state
                    # this also applies if you are defaulting today
                    # as the policy in that case is irrelevant.
                end
            end
        end
    end

    for iy in eachindex(y_grid)
        b_pol = model.zero_index
        for iyprime in eachindex(y_grid)
            push!(row_list, ny * nb + iy)
            push!(val_list, θ * T[iyprime, iy]) # reentry
            if alloc.repay[iyprime, b_pol]
                push!(col_list, ny * (b_pol - 1) + iyprime)
            else 
                push!(col_list, ny * nb + iyprime)  # default state
            end
            push!(row_list, ny * nb + iy)
            push!(val_list, (1 - θ) * T[iyprime, iy])  # stay out
            push!(col_list, ny * nb + iyprime)
        end
    end

    len = ny * (nb + 1)
    return dropzeros(sparse(col_list, row_list, val_list, len, len))
end


"""
    find_ergodic(T[, v0, max_iters, tol])

Given a transition probability T, and an initial vector v0, calculates
the ergodic distribution by pre-multiplying v0 by T until convergence. 
"""
function find_ergodic(
    T; 
    v0=ones(Float64, size(T)[1]) / size(T)[1], max_iters = _MAX_ITERS, tol = _TOL
)
    i = 1
    v1 = similar(v0)
    while true
        mul!(v1, T, v0)
        print_and_stop(chebyshev(v1, v0), i, tol, 100, max_iters) && break 
        i += 1
        v1, v0 = v0, v1
    end
    return v1
end 


"""
    find_ergodic(alloc::Allocation)

Returns the ergodic distribution associated with alloc.
Returns a matrix  of dimension ny * (nb + 1) containing the ergodic
distribution. There is an addition "debt" state (the last column), which 
represents the default state. 
"""
function find_ergodic(alloc::Allocation)
    @unpack nb, ny = alloc.model
    tt = create_transition_matrix(alloc)
    ergodic_1D = find_ergodic(tt)  # this Returns a 1-D vector ... 
    # .. and we transform it to a 2-D one.
    ergodic_2D = Array{Float64}(undef, ny, nb + 1) 
    for i in eachindex(ergodic_1D)
        # This exploits that the index of ergodic_1D is equivalent to the 
        # linear index in ergodic_2D (first iterate y_grid, then b_grid).
        ergodic_2D[i] = ergodic_1D[i]
    end 
    return ergodic_2D
end