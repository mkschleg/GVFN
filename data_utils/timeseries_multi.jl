#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH --mem=2000M # Memory request of 2 GB
#SBATCH --time=24:00:00 # Running time of 12 hours
#SBATCH --ntasks=1
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(joinpath(@__DIR__,".."))
using Distributed

@everywhere begin
    using Pkg
    Pkg.activate(joinpath(@__DIR__,".."))
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
end
parsed = parse_args(as)

# pids = addprocs(parsed["procs"]; exeflags="--project"
@everywhere using JLD2
@everywhere using FileIO
@everywhere using Statistics
@everywhere using ProgressMeter

# @everywhere begin

@everywhere begin
    
    function NRMSE(results, window, skip, subsample; phase)
        # Ground truth and predictions
        g,p = results["$(phase)GroundTruth"], results["$(phase)Predictions"]
        
        # drop the extra predictions past the last ground truth
        p = p[1:(size(g)[1]), :]
        
        # subsample the data
        # g, p = g[1:subsample:end], p[1:subsample:end]
        values = zeros(length(window+1:skip:(size(p)[1])), size(g)[2])
        
        for j ∈ 1:(size(g)[2])
            for (idx, i) = enumerate(window+1:skip:(size(p)[1]))
                nrm = mean((g[i-window:i, j].-mean(g[i-window:i, j])).^2)
                ĝ = g[i-window:i, j]
                P̂ = p[i-window:i, j]
                values[idx, j] = sqrt(mean((ĝ.-P̂).^2) / nrm)
            end
        end
        return values
    end

    TrainingNRMSE(results, window, skip, subsample) = NRMSE(results, window, skip, subsample; phase="")
    ValidationNRMSE(results, window, skip, subsample) = NRMSE(results, window, skip, subsample; phase="Validation")
    TestNRMSE(results, window, skip, subsample) = NRMSE(results, window, skip, subsample; phase="Test")
    
    loadData(itm) = try
        FileIO.load(joinpath(itm.folder_str,"results.jld2"))["results"]
    catch
        println(itm.folder_str)
        rethrow()
    end



    function inner_getNRMSE(itm, save_loc, window, skip, subsample)
        read_loc = itm.folder_str
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
        # functions = [TrainingNRMSE, ValidationNRMSE, TestNRMSE]
        functions = [TrainingNRMSE]
        pairs = Dict(Pair.(String.(Symbol.(functions)), functions))
        for (lbl, fn) ∈ pairs
            try
                nrmse = fn(loadData(itm), window, skip, subsample)
                results["$(lbl)"] = nrmse[:, 1]
            catch
            end
        end
        results["window"], results["skip"], results["subsample"] = window, skip, subsample

        if length(keys(results)) != 0
            JLD2.@save joinpath(tmp_save_loc, "results.jld2") results
            cp(joinpath(read_loc, "settings.jld2"), joinpath(tmp_save_loc, "settings.jld2"))
        end
    end 

end #end @everywhere
    
function getNRMSE()
    
    window, skip, subsample = parsed["window"], parsed["skip"], parsed["subsample"]

    read_dir = parsed["dir"][end] == '/' ? parsed["dir"][1:end-1] : parsed["dir"]
    fldr = split(read_dir,"/")[end]
    save_loc = joinpath(parsed["dest"],fldr*"_NRMSE")
    if !isdir(save_loc)
        mkdir(save_loc)
        mkdir(joinpath(save_loc, "data"))
    end
    ic = ItemCollection(joinpath(read_dir,"data"))

    println("Reading from: $(read_dir)")
    println("Saving to: $(save_loc)")
    @assert read_dir != save_loc

    @everywhere begin
        read_dir = $read_dir
        save_loc = $save_loc
        window, skip, subsample = $window, $skip, $subsample
    end
    
    @showprogress 0.1 "Parameter Settings: " pmap(ic.items) do itm
        inner_getNRMSE(itm, save_loc, window, skip, subsample)
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
