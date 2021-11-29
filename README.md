# One period defaultable bond model 

This presents the basic simulation of the one-period Eaton-Gersovitz model. See 
[Aguiar and Gopinath(2006)](https://linkinghub.elsevier.com/retrieve/pii/S0022199605000644) and 
[Arellano(2008)](https://www.aeaweb.org/articles?id=10.1257/aer.98.3.690).

For a more complete repository, including long-term bonds in the analysis, see: 

[The Economics of Sovereign Debt.](https://github.com/markaguiar/TheEconomicsofSovereignDebt)

All code is written for [Julia](https://julialang.org/).


## Installation

To install all necessary packages, open a julia prompt at the root of this repository and type:

    julia> using Pkg 
    julia> Pkg.activate(".")
    julia> Pkg.instantiate()

The above will download the packages needed.

The file `eaton_gersovitz.jl` contains the main code, with the type structure and methods to solve 
and simulate the model. 

See `example.jl` (or the accompanying jupyter notebook) for an example on how to call and use the code. 

