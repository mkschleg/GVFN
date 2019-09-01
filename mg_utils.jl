import Pkg; Pkg.activate(".")
using Revise
using Reproduce
using FileIO
using JLD2
using Statistics
using Plots; pyplot()

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
    "--seed", "1",
    "--alg", "BatchTD",
    "--steps", "600000",
    "--valSteps", "200000",
    "--testSteps", "200000",
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
    return results["GroundTruth"], results["Predictions"]
end

function plotData()
    g,p = getData()
    plot(p,  label="Predictions")
    plot!(g, label="Ground Truth")
end

function NRMSE()
    g,p = getData()
    p = p[1:length(g)]

    values = Float64[]
    n=10000
    for i=n+1:10:length(p)
        ĝ = g[i-n:i]
        P̂ = p[i-n:i]
        push!(values, sqrt(mean((ĝ.-P̂).^2) / mean((ĝ.-mean(ĝ)).^2)))
    end

    return values
end

function plotNRMSE()
    values = NRMSE()
    plot(values, ylim=[0,1],yticks=[0.1i for i=0:10], grid=false, label="NRMSE")
end
