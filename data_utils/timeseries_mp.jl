#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.4.1/bin/julia
#SBATCH --mem=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out
#SBATCH --account=rrg-whitem

using Pkg
# Pkg.activate(joinpath(@__DIR__,".."))
Pkg.activate("/project/6010404/mkschleg/GVFN")
using Distributed

@everywhere begin
    using Pkg
    # Pkg.activate(joinpath(@__DIR__,".."))
    Pkg.activate("/project/6010404/mkschleg/GVFN")
end

using Reproduce

as = ArgParseSettings()
@add_arg_table! as begin
    "--dir"
    arg_type=String
    "--dest"
    arg_type=String
    "--window"
    arg_type=Int
    "--skip"
    arg_type=Int
    "--subsample"
    arg_type=Int
    default=1
    "--numprocs"
    arg_type=Int
    default=1
end
parsed = parse_args(as)

pids = if parsed["numprocs"] != 1
    addprocs(parsed["numprocs"]-1; exeflags="--project")
end

@everywhere using JLD2
@everywhere using FileIO
@everywhere using Statistics
@everywhere using ProgressMeter


@everywhere begin
    function NRMSE(results, window, skip, subsample; phase)
        # Ground truth and predictions
        g,p = results["$(phase)GroundTruth"], results["$(phase)Predictions"]
        
        # drop the extra predictions past the last ground truth
        p = p[1:length(g)]

        # subsample the data
        g, p = g[1:subsample:end], p[1:subsample:end]
        
        values = Float64[]
        for i=window+1:skip:length(p)
            ĝ = g[i-window:i]
            P̂ = p[i-window:i]
            push!(values, sqrt(mean((ĝ.-P̂).^2) / mean((ĝ.-mean(ĝ)).^2)))
        end
        return values
    end

    TrainingNRMSE(results, window, skip, subsample) = NRMSE(results, window, skip, subsample; phase="")
    ValidationNRMSE(results, window, skip, subsample) = NRMSE(results, window, skip, subsample; phase="Validation")
    TestNRMSE(results, window, skip, subsample) = NRMSE(results, window, skip, subsample; phase="Test")

    loadData(itm) = FileIO.load(joinpath(itm.folder_str,"results.jld2"))["results"]
    loadData(d::String) = FileIO.load(joinpath(d,"results.jld2"))["results"]

    function inner_getNRMSE(read_loc, save_loc, window, skip, subsample)
        # read_loc = itm.folder_str
        read_file = joinpath(read_loc, "results.jld2")

        tmp_save_loc = joinpath(save_loc, "data", basename(read_loc))
        if !isdir(tmp_save_loc)
            mkpath(tmp_save_loc)
        else
            if isfile(joinpath(tmp_save_loc, "settings.jld2"))
                return;
            end
        end

        results = Dict()
        functions = [TrainingNRMSE, ValidationNRMSE, TestNRMSE]
        pairs = Dict(Pair.(String.(Symbol.(functions)), functions))
        for (lbl, fn) ∈ pairs
            nrmse = fn(loadData(read_loc), window, skip, subsample)
            results["$(lbl)"] = nrmse
        end
        results["window"], results["skip"], results["subsample"] = window, skip, subsample

        JLD2.@save joinpath(tmp_save_loc, "results.jld2") results
        cp(joinpath(read_loc, "settings.jld2"), joinpath(tmp_save_loc, "settings.jld2"))
    end
    
end

function getNRMSE()

    as = ArgParseSettings()
    @add_arg_table! as begin
        "--dir"
        arg_type=String
        "--dest"
        arg_type=String
        "--window"
        arg_type=Int
        "--skip"
        arg_type=Int
        "--subsample"
        arg_type=Int
        default=1
        "--numprocs"
        arg_type=Int
        default=1
    end
    parsed = parse_args(as)

    window, skip, subsample = parsed["window"], parsed["skip"], parsed["subsample"]

    read_dir = parsed["dir"][end] == '/' ? parsed["dir"][1:end-1] : parsed["dir"]
    fldr = split(read_dir,"/")[end]
    save_loc = joinpath(parsed["dest"],fldr*"_NRMSE")
    if !isdir(save_loc)
        mkdir(save_loc)
        mkdir(joinpath(save_loc, "data"))
    end

    dirs = readdir(joinpath(read_dir,"data"))
    # ic = if isfile(joinpath(read_dir, "settings_pre.jld2"))
    #     FileIO.load(joinpath(read_dir, "settings_pre.jld2"))["IC"]
    # else
    #     ItemCollection(joinpath(read_dir,"data"))
    # end

    println("Reading from: $(read_dir)")
    println("Saving to: $(save_loc)")
    @assert read_dir != save_loc

    @everywhere begin
        read_dir = $read_dir
        save_loc = $save_loc
        window, skip, subsample = $window, $skip, $subsample
    end
    
    @showprogress 0.1 "Parameter Settings: " pmap(dirs) do d
        inner_getNRMSE(joinpath(read_dir, "data", d), save_loc, window, skip, subsample)
    end

    cp(joinpath(read_dir, "settings"), joinpath(save_loc, "settings"))
    cp(joinpath(read_dir, "notes.org"), joinpath(save_loc, "notes.org"))
end

function moveData(src, dest)
    if !isdir(dest)
        mkpath(dest)
    end

    ic = ItemCollection(src)
    for itm ∈ ic.items
        if "normalizer"∈keys(itm.parsed_args)
            mv(itm.folder_str, joinpath(dest, split(itm.folder_str, "/")[end]))
        end
    end
end

getNRMSE()
