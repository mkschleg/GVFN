import Pkg; Pkg.activate(".")
using Revise
using Reproduce
using FileIO
using JLD2
using Statistics
using Plots; pyplot()

includet("experiment/timeseries.jl")
saveDir = "mackeyglass_bestGVFN"

# =============================
# --- D E B U G   U T I L S ---
# =============================

getArgs(seed) = [
    "--horizon", "12",
    "--batchsize", "32",
    "--model_stepsize", "0.001",
    "--model_opt", "ADAM",
    "--gvfn_stepsize", "3.0e-3",
    "--gvfn_opt", "Descent",
    "--gamma_high", "0.95",
    "--gamma_low", "0.2",
    "--num_gvfs", "128",
    "--seed", string(seed),
    "--alg", "BatchTD",
    "--steps", "600000",
    "--valSteps", "200000",
    "--testSteps", "200000",
    "--exp_loc", saveDir,
    "--env", "MackeyGlass"
]

exp() = exp(4)
exp(seed::Int) = exp(getArgs(seed))
exp(args) = TimeSeriesExperiment.main_experiment(args)

# ===============
# --- D A T A ---
# ===============

function getResults()
    ic = ItemCollection(saveDir)
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

function NRMSE(hashes)
    all_values = Vector{Float64}[]
    for idx = 1:length(hashes)
        h = hashes[idx]
        f = joinpath(h,"results.jld2")
        if !isfile(f)
            return [Inf]
        end
        results = load(f,"results")

        g,p = results["GroundTruth"], results["Predictions"]

        p = p[1:length(g)]

        values = Float64[]
        n=10000
        for i=n+1:10:length(p)
            ĝ = g[i-n:i]
            P̂ = p[i-n:i]
            push!(values, sqrt(mean((ĝ.-P̂).^2) / mean((ĝ.-mean(ĝ)).^2)))
        end
        push!(all_values, values)
    end

    if length(all_values)==0
        return [Inf]
    end

    vals = zeros(length(all_values),length(all_values[1]))
    for i=1:length(all_values)
        vals[i,:] .= all_values[i]
    end
    return vals
end

function getBestNRMSE()
    ic = ItemCollection(saveDir)
    d = diff(ic)
    delete!(d, "seed")
    iter = Iterators.product(values(d)...)
    best = Inf
    bestData = nothing
    for arg in iter
        # length(hashes) == number of seeds
        _, hashes, _ = search(ic, Dict(Pair.(keys(d), arg)))

        # AUC
        data = NRMSE(hashes)
        value = mean(data)
        if value<best
            best = value
            bestData = data
        end
    end

    @assert bestData != nothing
    return bestData
end

function plotNRMSE()
    values = getBestNRMSE()
    av = mean(values, dims=1)
    σ = std(values, dims=1, corrected=true) / sqrt(size(values,1))
    plot(av', ribbon=σ', ylim=[0,1],yticks=[0.1i for i=0:10], grid=false, label="NRMSE")
end

function plotData(b::Dict)
    p=plot()
    plot!(b["Predictions"])
    plot!(b["GroundTruth"])
    plot(p, ylim=[0,2])
end
