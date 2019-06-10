#!/cvmfs/soft.computecanada.ca/easybuild/software/2017/avx2/Compiler/gcc7.3/julia/1.1.0/bin/julia
#SBATCH --mem=2000M # Memory request of 2 GB
#SBATCH --time=12:00:00 # Running time of 12 hours
#SBATCH --ntasks=1
#SBATCH --account=rrg-whitem

using Pkg
Pkg.activate(".")

using Reproduce
using JLD2
using FileIO
using Statistics
using ProgressMeter


function main_cycleworld_rnn()

    as = ArgParseSettings()
    @add_arg_table as begin
        "readdir"
        arg_type=String
    end
    parsed = parse_args(as)

    read_dir = parsed["readdir"]

    println("Reading from: $(read_dir)")

    save_loc = read_dir*"_clean"
    if !isdir(save_loc)
        mkdir(save_loc)
    end
    ic = ItemCollection(read_dir)
    diff_dict = diff(ic.items)
    dict_keys = collect(keys(diff_dict))

    args = Iterators.product([diff_dict[key] for key in dict_keys]...)
    # "out_err_strg"=>out_err_strg])
    @showprogress 0.1 "Args: " for arg_tuple in collect(args)
        search_dict = Dict([key=>arg_tuple[idx] for (idx, key) in enumerate(collect(dict_keys))])
        _, read_hash_list, _ = search(ic, search_dict)
        read_loc = read_hash_list[1]
        read_file = joinpath(read_loc, "results.jld2")

        tmp_save_loc = joinpath(save_loc, basename(read_loc))
        if !isdir(tmp_save_loc)
            mkdir(tmp_save_loc)
        else
            if isfile(joinpath(tmp_save_loc, "settings.jld2"))
                continue;
            end
        end

        results = FileIO.load(read_file)
        old_results = results
        new_results = mean(old_results["out_err_strg"])
        new_results_early = 0.0# mean(old_results["out_err_strg"][1:100000])
        new_results_end = 0.0# mean(old_results["out_err_strg"][250000:end])
        results = Dict(["mean"=>new_results, "mean_early"=>new_results_early, "mean_end"=>new_results_end])

        
        @save joinpath(tmp_save_loc, "results.jld2") results
        cp(joinpath(read_loc, "settings.jld2"), joinpath(tmp_save_loc, "settings.jld2"))
    end

    cp(joinpath(read_dir, "settings"), joinpath(save_loc, "settings"))
    cp(joinpath(read_dir, "notes.org"), joinpath(save_loc, "notes.org"))
end

main_cycleworld_rnn()


