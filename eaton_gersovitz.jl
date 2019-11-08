# This code solves the one-period Eaton-Gersovitz model with 
# one-period debt allowing for Epstein-Zin recursive utility. 

using Parameters
using PyPlot
using SpecialFunctions
# using Distances
using LinearAlgebra

const _TOL = 10.0^(-12)
const _MAX_ITERS = 10000 


########################################################################
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
    α = 2.0 # 1/IES parameter
    θ = 0.282   # prob of regaining market access
    b_max = 1.2   # maximum debt level
    b_min = -0.2   # minimum debt level 
    nb_approx::Int64 = 200  # approximate points for B grid 
    y_process::M = logAR1(
        N=100, 
        ρ=0.948503, 
        σ=0.027093, 
        μ=0.0, 
        span=3.0, 
        inflate_ends=false
    ) 
    # y_process should be a named tuple with a transition matrix, T, a grid, grid,
    # as well as a value for the ergodic mean, mean. 
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
    d_and_c_fun::F2 = get_d_and_c_fun(nb)
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


#
# Functions related to the constructor of TwoStateModel
#


# from here: https://stackoverflow.com/questions/25678112/insert-item-into-a-sorted-list-with-julia-with-and-without-duplicates
insert_and_dedup!(v::Vector, x) = (splice!(v, searchsorted(v,x), x); v)


function generate_b_grid(b_min, b_max, nB_approx)
    grid = collect(range(b_min, stop=b_max, length=nB_approx))
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

# Taken from Stelios code 
function  logAR1(;
    N=100, ρ=0.948503, σ=0.027093, μ=0.0, span=3.0, inflate_ends=true
)
    # Get discretized space using Tauchen's method

    # Define versions of the standard normal cdf for use in the tauchen method
    std_norm_cdf(x::Real) = 0.5 * erfc(-x/sqrt(2))
    std_norm_cdf(x::AbstractArray) = 0.5 .* erfc(-x./sqrt(2))

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
    return (T=T, grid=y_grid, mean=y_mean)
end


function get_d_and_c_fun(gridlen::Int64)
    """
    Generate a bisection tree from array to be used in the "divide and conquer"
    algorithm. 

    Returns a tuple of two arrays. First element is the list of the elements in
    array, excluding the extrema. Second element is an array of tupples with the
    each of the parents.
    """
    function create_tree(array)

        #    Auxiliary function
        function create_subtree!(
            tree_list, 
            parents_list, # modified in place
            array
        )
            length = size(array)[1]
            if length == 2
                return
            else
                parents = (array[1], array[end])
                halflen = (length + 1)÷2
                push!(tree_list, array[halflen])
                push!(parents_list, parents)
                create_subtree!(
                    tree_list, parents_list, @view array[1:halflen]
                )
                create_subtree!(
                    tree_list, parents_list, @view array[halflen:end]
                )
            end 
        end

        tree_list = eltype(array)[]
        parents_list = Tuple{eltype(array), eltype(array)}[]
        create_subtree!(tree_list, parents_list, array)
        return (tree_list, parents_list)
    end

    # Given an index i and a tree returns the current index in the three as well 
    # as the indices of the parents.
    function d_and_c_index_bounds(i, pol_vector, tree)
        if i == 1 
            b_i, left_bound, right_bound = 1, 1, gridlen
        elseif i == 2
            b_i, left_bound, right_bound  = gridlen, pol_vector[1], gridlen 
        else
            index = i - 2
            b_i = tree[1][index]
            left_bound = pol_vector[tree[2][index][1]]
            right_bound = pol_vector[tree[2][index][2]]
        end 
        return (b_i, left_bound, right_bound)
    end

    tree = create_tree(collect(1:gridlen))
    return ((x, pol) -> d_and_c_index_bounds(x, pol, tree))
end


function EZutility(model, c, V)
    @unpack β, u, γ, α = model 
    tmp = (1-β) * u(c) + β * V^((1-γ)/(1-α))
    return tmp^(1/(1-γ))
end


function update_vD!(new, model, old; tmp=similar(new.vD))
    @unpack T, θ, zero_index, y_def_grid, y_grid, α = model
    
    mm = @view new.vMax[:, zero_index] # vMax allows the possibility of 
        # immediate default after re-entry. Important for iterations away from
        # fixed point. 
    tmp .= θ .* (mm.^(1-α)) .+ (1-θ) .* old.vD.^(1-α)
    Threads.@threads for iy in eachindex(y_grid)
        cont_value = 0.0 
         for iy′ in eachindex(y_grid)
            cont_value += T[iy′, iy] * tmp[iy′]
         end 
         new.vD[iy] = EZutility(model, y_def_grid[iy], cont_value)
    end
end


function update_q!(new, model)
    @unpack T, R = model
    mul!(new.q, T', new.repay)
    new.q .*= R^(-1)
end


function assign!(new, iy, ib, vR, vMax, repay, b_pol)
    new.vR[iy, ib] = vR
    new.vMax[iy, ib] = vMax
    new.repay[iy, ib] = repay
    new.b_pol[iy, ib] = b_pol
end


function solve_for_single_iy!(new, tmp_EV, tmp_vMax, model, old, iy)
    @unpack u, β, γ, α, y_grid, b_grid, d_and_c_fun, T, nb, ny, min_v = model

    # precomputing the continuation value
     
    # for ib in 1:nb
    #     acc = 0.0
    #     for iyprime in 1:ny
    #         acc += T[iyprime, iy] * tmp_vMax[iyprime, ib]
    #     end
    #     tmp_EV[iy, ib] = β * acc^((1-γ)/(1-α))
    # end
    default_at = nb + 1
    for ib_iter in 1:nb
        ib, left_bound, right_bound = d_and_c_fun(
            ib_iter, 
            @view new.b_pol[iy, :]
        )
        if ib > default_at 
            # already defaulted with less debt -- default here too
            assign!(new, iy, ib, min_v, old.vD[iy], false, 0)
        else
            first_valid = true 
            current_max = min_v 
            policy = 0
            for ib_prime in left_bound:right_bound
                c = y_grid[iy]+ old.q[iy, ib_prime] * b_grid[ib_prime] - 
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
                assign!(new, iy, ib, current_max, current_max, true, policy)
            else 
                # not feasible or default is preferred
                assign!(new, iy, ib, current_max, old.vD[iy], false, policy)
                default_at = ib
            end 
        end
    end
end


function iterate_once!(
    new, model, old; 
    tmp_EV=similar(new.vR), tmp_vMax=similar(new.vR), tmp_EvD=similar(new.vD)
)
    @unpack T, β, γ, α = model

    # auxiliary calculation of expected continuation value function
    tmp_vMax .= old.vMax.^(1-model.α)  
    mul!(tmp_EV, T', tmp_vMax)
    tmp_EV .= β .* tmp_EV .^ ((1-γ)/(1-α))

    Threads.@threads for iy in 1:model.ny
        solve_for_single_iy!(new, tmp_EV, tmp_vMax, model, old, iy)
    end 
    update_vD!(new, model, old; tmp=tmp_EvD)
    update_q!(new, model)
end


# Helper distance function 
function distance(new, old)
    error = 0.0
    for (a, b, c, d) in zip(new.vR, old.vR, new.q, old.q)
        error = max(error, abs(a - b), abs(c - d))
    end 
    return error
end


function solve(
    model; 
    new=Allocation(model), 
    start=Allocation(model),
    max_iters::Int64=_MAX_ITERS,
    tol::Float64=_TOL
)
    tmp1, tmp2, tmp3 = similar(new.vR), similar(new.vR), similar(new.vD)
    old = start
    dist = 0.0
    for i in 1:max_iters
        iterate_once!(
            new, model, old; 
            tmp_EV=tmp1, tmp_vMax=tmp2, tmp_EvD=tmp3
        )
        dist = distance(new, old)
        # dist = chebyshev(new.vR, old.vR) + chebyshev(new.q, old.q)
        if mod(i, 100) == 1
            println("Iteration:", i, "  error:", dist)
        end 
        if (dist < tol)
            println("Iteration:", i, " error: ", dist)
            return new
        else
            new, old = old, new
        end
    end
    println("Did not converge!, iteration: ", max_iters, " error: ", dist)
    return new
end

