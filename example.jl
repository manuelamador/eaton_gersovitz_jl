# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Julia 4 threads 1.6.3
#     language: julia
#     name: julia-4-threads-1.6
# ---

# # Eaton Gersovitz Example 

# A simple example of how to use the code. 

import Pkg 
Pkg.activate(@__DIR__)
Pkg.instantiate()

using Plots
default(label = "", lw = 2)

using Revise
includet(joinpath(@__DIR__, "eaton_gersovitz.jl"))

m = EatonGersovitzModel(β = 0.953)

@time sol = solve(m);

plot(sol.q[100, :])

plot(sol.b_pol[100, :])
plot!(sol.b_pol[100, :], sol.b_pol[100, :], ls = :dash)

dist = find_ergodic(sol);

plot(sum(dist, dims = 1)[1:end-1])

sum(dist, dims = 1)[end]


