
using Pkg
Pkg.activate(".")
include("experiment/cycleworld.jl")

CycleWorldExperiment.main_experiment(ARGS)
