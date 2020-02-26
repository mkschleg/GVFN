using Revise
using Reproduce
using FileIO
using JLD2
using Statistics
using Plots; gr()
using Reproduce.Config

includet("experiment/timeseries.jl")

const default_config = "configs/test_gvfn.toml"
const saveDir = string(@__DIR__)

# =============================
# --- D E B U G   U T I L S ---
# =============================

function exp(cfg, idx, run=1)
    cfg = ConfigManager(cfg, saveDir)
    parse!(cfg, idx, run)
    TimeSeriesExperiment.main_experiment(cfg; progress=true)
end

exp(run) = exp(default_config, 1, run)
exp() = exp(default_config, 1)

# ===============
# --- D A T A ---
# ===============

function getResults()
    cfg = ConfigManager(default_config, saveDir)
    parse!(cfg,1,1)
    return get_run_data(cfg,1,1)
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

function NRMSE(cfg, idx)
    parse!(cfg, idx)
    nruns = cfg["args"]["nruns"]

    all_values = Vector{Float64}[]
    for r=1:nruns
        results = get_run_data(cfg, idx, r)

        g,p = results["GroundTruth"], results["Predictions"]

        p = p[1:length(g)]

        values = Float64[]
        start = 10000
        for i=start+1:10:length(p)
            ĝ = g[i-start:i]
            P̂ = p[i-start:i]
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
    cfg = ConfigManager(default_config, saveDir)

    best = Inf
    bestData = nothing
    bestIdx = nothing
    for idx=1:total_combinations(cfg)
        data = NRMSE(cfg, idx)
        value = mean(data)
        if value<best
            best = value
            bestData = data
            bestIdx = idx
        end
    end
    @assert bestData != nothing

    println("best parameter index: $(bestIdx)")
    return bestData
end

function plotNRMSE()
    values = getBestNRMSE()
    av = mean(values, dims=1)
    σ = std(values, dims=1, corrected=true) / sqrt(size(values,1))
    plot(av', ribbon=σ', grid=false, label="NRMSE",ylim=[0,2])
end

function plotData(b::Dict)
    p=plot()
    plot!(b["Predictions"])
    plot!(b["GroundTruth"])
    plot(p, ylim=[0,2])
end

