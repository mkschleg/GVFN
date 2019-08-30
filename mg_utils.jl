import Pkg; Pkg.activate(".")
using Revise
using Reproduce
using FileIO
using JLD2
using Plots

includet("experiment/mackeyglass.jl")

# =============================
# --- D E B U G   U T I L S ---
# =============================

getArgs() = [
    "--horizon", "12",
    "--batchsize", "32",
    "--model_stepsize", "0.001",
    "--model_opt", "ADAM",
    "--gvfn_stepsize", "3.0e-5",
    "--gvfn_opt", "Descent",
    "--gamma_high", "0.95",
    "--gamma_low", "0.2",
    "--num_gvfs", "128",
    "--seed", "4",
    "--alg", "BatchTD",
    "--steps", "6000",
    "--valSteps", "2000",
    "--testSteps", "2000",
    "--exp_loc", "mackeyglass_gvfn"
]

exp() = exp(getArgs())
exp(args) = MackeyGlassExperiment.main_experiment(args)

# ===============
# --- D A T A ---
# ===============

function getResults()
    ic = ItemCollection("mackeyglass_gvfn")
    _,hashes,_ = search(ic, Dict())
    return load(joinpath(hashes[1], "results.jld2"), "results")
end

function getData()
    results = getResults()
    return results["GroundTruth"], vcat(results["Predictions"], results["ValidationPredictions"], results["TestPredictions"])
end
