import Pkg; Pkg.activate(".")
using Revise
using Reproduce
using FileIO
using JLD2
using Statistics
using Plots; pyplot()

includet("experiment/timeseries.jl")

const env = "MackeyGlass"
const saveDir = "GVFN_TEST"

# =============================
# --- D E B U G   U T I L S ---
# =============================

getArgs(seed) = [
    "--seed", string(seed),
    "--alg", "BatchTD",
    "--steps", "600000",
    "--valSteps", "200000",
    "--testSteps", "200000",
    "--exp_loc", saveDir,
    "--env", env,

    # BPTT

    # GVFN
    "--horizon", "12",
    "--batchsize", "32",
    "--model_stepsize", "0.001",
    "--model_opt", "ADAM",
    "--gvfn_stepsize", "3.0e-3",
    "--gvfn_opt", "Descent",
    "--gvfn_tau", "4",
    "--gamma_high", "0.95",
    "--gamma_low", "0.2",
    "--num_gvfs", "128",

    # RNN
    "--rnn_opt", "ADAM",
    "--rnn_tau", "4",
    "--rnn_lr", "0.001",
    "--rnn_nhidden", "32",
    "--rnn_cell", "GRU",

    "--agent", "GVFN"

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
    bestHashes = nothing
    for arg in iter
        # length(hashes) == number of seeds
        _, hashes, _ = search(ic, Dict(Pair.(keys(d), arg)))

        # AUC
        data = NRMSE(hashes)
        value = mean(data)
        if value<best
            best = value
            bestData = data
            bestHashes = hashes
        end
    end
    @assert bestData != nothing

    println("best hashes:")
    @show bestHashes

    return bestData
end

function plotNRMSE()
    values = getBestNRMSE()
    av = mean(values, dims=1)
    σ = std(values, dims=1, corrected=true) / sqrt(size(values,1))
    plot(av', ribbon=σ', grid=false, label="NRMSE")
end

function plotData(b::Dict)
    p=plot()
    plot!(b["Predictions"])
    plot!(b["GroundTruth"])
    plot(p, ylim=[0,2])
end

# function synopsis_rnn(exp_loc::String; best_args=["truncation", "cell"])
#     # Iterators.product
#     args = Iterators.product(["mean", "median", "best"], ["all", "end"])
#     func_dict = Dict(
#         "all"=>cycleworld_data_clean_func_rnn,
#         "end"=>cycleworld_data_clean_func_rnn_end)

#     if !isdir(joinpath(exp_loc, "synopsis"))
#         mkdir(joinpath(exp_loc, "synopsis"))
#     end

#     for a in args
#         @info "Current Arg $(a)"
#         order_settings(
#             exp_loc;
#             run_key="seed",
#             clean_func=func_dict[a[2]],
#             runs_func=runs_func,
#             sort_idx=a[1],
#             save_locs=[joinpath(exp_loc, "synopsis/order_settings_$(a[1])_$(a[2]).$(ext)") for ext in ["jld2", "txt"]])
#     end

#     ret = best_settings(exp_loc, best_args;
#                         run_key="seed", clean_func=cycleworld_data_clean_func_rnn,
#                         runs_func=runs_func,
#                         sort_idx="mean",
#                         save_locs=[joinpath(exp_loc, "best_trunc_horde.txt")])

#     ret = best_settings(exp_loc, best_args;
#                         run_key="seed", clean_func=cycleworld_data_clean_func_rnn_end,
#                         runs_func=runs_func,
#                         sort_idx="mean",
#                         save_locs=[joinpath(exp_loc, "best_trunc_horde_end.txt")])
# end
