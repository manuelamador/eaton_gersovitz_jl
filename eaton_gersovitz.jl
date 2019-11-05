# This is the code to solve the one-period Eaton-Gersovitz model with 
# one-period debt. 
#
# It allows for Epstein-Zin recursive utility. 

using Parameters
using PyPlot
using SpecialFunctions

const _TOL = 10.0^(-12)
const _MAX_ITERS = 10000 


########################################################################
#
# Structs and constructors
# 
#

struct MarkovChain{T1, T2}
    T::Array{T1, 2}
    grid::Array{T2, 1}
    mean::T2 
end


@with_kw struct AR1InLogs @deftype Float64  
    N::Integer = 100
    ρ = 0.948503
    σ = 0.027093
    μ = 1.0
    span = 3.0
    inflate_ends::Bool = true
end 


# `EatonGersovitz` contains the basic parameters of the model as well 
# as some other values and the debt grid. 
@with_kw struct EatonGersovitzModel{F1, F2} @deftype Float64
    R = 1.05 
    β = 0.91 
    γ = 2.0 # risk aversion parameter
    α = 2.0 # 1/IES parameter
    ρ = 0.948503  # persistence in output
    η = 0.027093  # st dev of output shocks 
    θ = 0.282   # prob of regaining market access
    ny = 100  # points for y grid 
    nB = 101  # points for B grid 
    debt_max = 0.45   # maximum debt level
    debt_min = -0.1   # minimum debt level 
    b_grid = collect(range(debt_min, stop=debt_max, length=nB))
    y_chain::MarkovChain = discretize(AR1InLogs(
        N=100, ρ=0.948503, σ=0.027093, μ=0.0, span=3.0, inflate_ends=true
        ))
    y_grid::Array{Float64, 1} = y_chain.grid
    T::Array{Float64, 2} = y_chain.T
    y_mean = y_chain.mean
    y_def_fun::F1 = arellano_default_cost(0.969, y_mean)
    y_def_grid::Array{Float64, 1} = y_def_fun.(y_grid)
    d_and_c_fun::F2 = get_d_and_c_fun(gridlen)
end


function Base.show(io::IO, model::EatonGersovitzModel)
    @unpack R, β, τH, τL, λ, δ, y, gridlen = model
    print(
        io, "R=", R, " β=", β
    )    
end


# `Alloc` stores an allocation together with a reference to the model that 
# generated it. 
struct Eqm{P}
    v::Array{Float64, 2}  # repayment value function
    q::Array{Float64, 2}  # price value function 
    b_pol_i::Array{Int64, 2}   # policy function 
    model::P    # reference to the parent model
end 


Eqm(model::EatonGersovitzModel) = Eqm(
    fill(NaN, model.gridlen),
    fill(NaN, model.gridlen),
    fill(NaN, model.gridlen),
    zeros(Int64, model.gridlen),
    model
)

function Base.show(io::IO, eqm::Eqm)
    @unpack R, β, τH, τL, λ, δ, y, gridlen = eqm.model
    print(io, "Eqm for model: ")
    show(io, eqm.model)   
end


#
# Functions related to the constructor of TwoStateModel
#


# from here: https://stackoverflow.com/questions/25678112/insert-item-into-a-sorted-list-with-julia-with-and-without-duplicates
insert_and_dedup!(v::Vector, x) = (splice!(v, searchsorted(v,x), x); v)


function arellano_default_cost(par, meanx)
    return x -> min(x, par * meanx)
end 


function proportional_default_cost(par)
    return x -> par * x
end 


function quadratic_default_cost(h1, h2)
    return x -> (x - max(zero(h1), h1 * x + h2 * x^2))
end 


function discretize(g::AR1InLogs)
    # Get discretized space using Tauchen's method

    # Define versions of the standard normal cdf for use in the tauchen method
    std_norm_cdf(x::Real) = 0.5 * erfc(-x/sqrt(2))
    std_norm_cdf(x::AbstractArray) = 0.5 .* erfc(-x./sqrt(2))

    @unpack ρ, σ, μ, N, span, inflate_ends = g
    a_bar = span * σ * sqrt(1 - ρ^2)
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
        for i = 1:N
            for j = 1:N
                T[j, i] = (std_norm_cdf((y[j] - ρ*y[i] + d/2) / σ) -
                    std_norm_cdf((y[j] - ρ*y[i] - d/2) / σ))
            end
        end
    end
    y_grid = exp.(y .+ μ / (1 - ρ))
    y_mean = exp(μ / (1 - ρ) + 1/2 * σ^2 / (1- ρ^2))
    T_sums = sum(T, dims=1)
    for i in 1:N
        T[:,i] .*= T_sums[i]^(-1)
    end
    return MarkovChain(T, y_grid, y_mean)
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



