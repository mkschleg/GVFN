#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH --mem=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=1
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(joinpath(@__DIR__,".."))

using Reproduce
using JLD2
using FileIO
using Statistics
using ProgressMeter

function NRMSE(results; phase="Validation", window, skip)
    # Ground truth and predictions
    g,p = results["$(phase)GroundTruth"], results["$(phase)Predictions"]

    # drop the extra predictions past the last ground truth
    p = p[1:length(g)]

    values = Float64[]
    for i=window+1:skip:length(p)
        ĝ = g[i-window:i]
        P̂ = p[i-window:i]
        push!(values, sqrt(mean((ĝ.-P̂).^2) / mean((ĝ.-mean(ĝ)).^2)))
    end
    return values
end

TrainingNRMSE(results, window, skip) = NRMSE(results; phase="",window=window,skip=skip)
ValidationNRMSE(results, window, skip) = NRMSE(results; phase="Validation",window=window,skip=skip)
TestNRMSE(results, window, skip) = NRMSE(results; phase="Test",window=window,skip=skip)

loadData(itm) = load(joinpath(itm.folder_str,"results.jld2"))["results"]

function get_setting(ic, clean_func; window, skip)

    diff_dict = diff(ic)

    itms = ic.items
    res = Vector{Float64}[]
    for (itm_idx, itm) ∈ enumerate(itms)
        push!(res, clean_func(loadData(itm), window, skip))
    end

    vals = zeros(length(res[1]),length(res))
    for i=1:length(res)
        vals[:,i] .= res[i]
    end

    μ = mean(vals,dims=2)
    σ = std(vals, dims=2, corrected=true)./sqrt(length(itms))

    return μ, σ
end

function getNRMSE()

    as = ArgParseSettings()
    @add_arg_table! as begin
        "--dir"
        arg_type=String
        "--dest_dir"
        arg_type=String
        default="."
        "--window"
        arg_type=Int
        "--skip"
        arg_type=Int
    end
    parsed = parse_args(as)

    read_dir = parsed["dir"][end] == '/' ? parsed["dir"][1:end-1] : parsed["dir"]
    window = parsed["window"]
    skip = parsed["skip"]


    fldr = split(read_dir,"/")[end]
    save_loc = joinpath(parsed["dest_dir"],fldr*"_NRMSE")
    if !isdir(save_loc)
        mkdir(save_loc)
    end
    ic = ItemCollection(joinpath(read_dir,"data"))

    println("Reading from: $(read_dir)")
    println("Saving to: $(save_loc)")
    @assert read_dir != save_loc

    @showprogress 0.1 "Parameter Settings: " for itm ∈ ic.items
        read_loc = itm.folder_str
        read_file = joinpath(read_loc, "results.jld2")

        tmp_save_loc = joinpath(save_loc, "data",basename(read_loc))
        if !isdir(tmp_save_loc)
            mkpath(tmp_save_loc)
        else
            if isfile(joinpath(tmp_save_loc, "settings.jld2"))
                continue;
            end
        end

        results = Dict()
        functions = [TrainingNRMSE, ValidationNRMSE, TestNRMSE]
        pairs = Dict(Pair.(String.(Symbol.(functions)), functions))
        for (lbl, fn) ∈ pairs
            nrmse = fn(loadData(itm), window, skip)
            results["$(lbl)"] = nrmse
        end
        results["window"], results["skip"] = window, skip

        @save joinpath(tmp_save_loc, "results.jld2") results
        cp(joinpath(read_loc, "settings.jld2"), joinpath(tmp_save_loc, "settings.jld2"))
    end

    cp(joinpath(read_dir, "settings"), joinpath(save_loc, "settings"))
    cp(joinpath(read_dir, "notes.org"), joinpath(save_loc, "notes.org"))
end

getNRMSE()
