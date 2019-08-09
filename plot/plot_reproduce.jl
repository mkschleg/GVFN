using Plots
using Reproduce
using Statistics
using ProgressMeter
using FileIO
using JLD2

# These functions are for grid searches.


function save_settings(save_loc, settings_vec)
    if split(basename(save_loc), ".")[end] == "txt"
        open(save_loc, "w") do f
            for v in settings_vec
                write(f, string(v)*"\n")
            end
        end
    else
        @save save_loc Dict("settings"=>settings_vec)
    end
end


"""
    best_settings

This function takes an experiment directory and finds the best setting for the product of arguments
with keys specified by product_args. To see a list of viable arguments use
    `ic = ItemCollection(exp_loc); diff(ic.items)`

If a save_loc is provided, this will save to the file specified. The fmt must be supported by FileIO and be able to take dicts.

Additional kwargs are passed to order_settings.

"""


function best_settings(exp_loc, product_args::Vector{String};
                       save_locs=nothing, kwargs...)

    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)

    args = Iterators.product([diff_dict[arg] for arg in product_args]...)
    # settings_vec = Vector{Tuple{Float64, Dict}}(undef, length(args))
    settings_dict = Dict()
    for (arg_idx, arg) in enumerate(args)
        search_dict = Dict([product_args[idx]=>key for (idx, key) in enumerate(arg)]...)
        ret = order_settings(exp_loc; set_args=search_dict, ic=ic, kwargs...)
        settings_dict[search_dict] = ret[1]
    end

    
    if save_locs != nothing
        # Save data
        # save_settings(save_loc, settings_dict)
        if typeof(save_locs) <: AbstractString
            save_settings(save_locs, settings_dict)
        elseif typeof(save_locs) <: AbstractArray
            for save_loc in save_locs
                save_settings(save_loc, settings_dict)
            end
        end
    else
        return settings_dict
    end
    
end


"""
    order_settings

    This provides a mechanism to order the settings of an experiment.

    kwargs:
        `set_args(=Dict{String, Any}())`: narrowing the search parameters. See best_settings for an example of use.

        `clean_func(=identity)`: The function used to clean the loaded data
        `runs_func(=mean)`: The function which takes a vector of floats and produces statistics. Must return either a Float64 or Dict{String, Float64}. (WIP, any container/primitive which implements get_index).

        `lt(=<)`: The less than comparator.
        `sort_idx(=1)`: The idx of the returned `runs_func` structure used for sorting.
        `run_key(=run)`: The key used to specify an ind run for an experiment.

        `results_file(=\"results.jld2\")`: The string of the file containing experimental results.
        `save_loc(=\"\")`: The save location (returns settings_vec if not provided).
        `ic(=ItemCollection([])`: Optional item_collection, not needed in normal use.
        
"""



function order_settings(exp_loc;
                        results_file="results.jld2",
                        clean_func=identity, runs_func=mean,
                        lt=<, sort_idx=1, run_key="run",
                        set_args=Dict{String, Any}(),
                        ic=ItemCollection([]), save_locs=nothing)

    if exp_loc[end] == '/'
        exp_loc = exp_loc[1:end-1]
    end

    exp_path = dirname(exp_loc)
    if length(ic.items) == 0
        ic = ItemCollection(exp_loc)
    end
    diff_dict = diff(ic.items)
    product_args = collect(filter((k)->(k!=run_key && k∉keys(set_args)), keys(diff_dict)))
    # println(product_args)
    args = Iterators.product([diff_dict[arg] for arg in product_args]...)

    settings_vec =
        Vector{Tuple{Union{Float64, Dict{String, Float64}}, Dict{String, Any}}}(undef, length(args))
    #####
    # Populate settings Vector
    #####
    # @info "Populating settings vector"
    @showprogress 0.1 "Setting: " for (arg_idx, arg) in enumerate(args)
        # settings_vec[arg_idx]
        search_dict = merge(
            Dict([product_args[idx]=>key for (idx, key) in enumerate(arg)]...),
            set_args)
        _, hashes, _ = search(ic, search_dict)
        μ_runs = zeros(length(hashes))
        for (idx_d, d) in enumerate(hashes)
            # try
            # println(joinpath(exp_path, d, results_file))
            if isfile(joinpath(exp_path, d, results_file))
                results = load(joinpath(exp_path, d, results_file))
                μ_runs[idx_d] = clean_func(results)
            else
                μ_runs[idx_d] = Inf
            end
        end
        settings_vec[arg_idx] = (runs_func(μ_runs), search_dict)
    end

    #####
    # Sort settings vector
    #####

    sort!(settings_vec; lt=lt, by=(tup)->tup[1][sort_idx])
    
    if save_locs != nothing
        if typeof(save_locs) <: AbstractString
            save_settings(save_locs, settings_vec)
        elseif typeof(save_locs) <: AbstractArray
            for save_loc in save_locs
                save_settings(save_loc, settings_vec)
            end
        end
    else
        return settings_vec
    end
end


function sensitivity(exp_loc, sweep_arg::String, product_args::Vector{String};
                     results_file="results.jld2", clean_func=identity,
                     ci_const = 1.96, sweep_args_clean=identity, save_dir="sensitivity", ylim=nothing)

    gr()

    if exp_loc[end] == '/'
        exp_loc = exp_loc[1:end-1]
    end
    head_dir = dirname(exp_loc)

    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)
    args = Iterators.product([diff_dict[arg] for arg in product_args]...)

    p1 = ProgressMeter.Progress(length(args), 0.1, "Args: ", offset=0)

    for arg in args

        plt=nothing
        μ = zeros(length(diff_dict[sweep_arg]))
        σ = zeros(length(diff_dict[sweep_arg]))

        p2 = ProgressMeter.Progress(length(diff_dict[sweep_arg]), 0.1, "$(sweep_arg): ", offset=1)
        for (idx, s_a) in enumerate(diff_dict[sweep_arg])
            search_dict = Dict(sweep_arg=>s_a, [product_args[idx]=>key for (idx, key) in enumerate(arg)]...)
            _, hashes, _ = search(ic, search_dict)
            # println(search_dict)
            # println(length(hashes))
            μ_runs = zeros(length(hashes))
            for (idx_d, d) in enumerate(hashes)

                if isfile(joinpath(head_dir, d, results_file))
                    results = load(joinpath(head_dir, d, results_file))
                    μ_runs[idx_d] = clean_func(results)
                # catch e
                else
                    # println(joinpath(head_dir, d, results_file))
                    μ_runs[idx_d] = Inf
                end

            end
            μ[idx] = mean(μ_runs)
            # println(μ)
            σ[idx] = ci_const * std(μ_runs)/sqrt(length(μ_runs))
            next!(p2)
        end

        if plt == nothing
            plt = plot(sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ, ylim=ylim)
        else
            plot!(plt, sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ)
        end

        if !isdir(joinpath(exp_loc, save_dir))
            mkdir(joinpath(exp_loc, save_dir))
        end

        save_file_name = join(["$(key)_$(arg[idx])" for (idx, key) in enumerate(product_args)], "_")

        savefig(plt, joinpath(exp_loc, save_dir, "$(save_file_name).pdf"))
        next!(p1)
    end


end


function sensitivity_multiline(exp_loc, sweep_arg::String, line_arg::String, product_args::Vector{String};
                               results_file="results.jld2", clean_func=identity,
                               sweep_args_clean=identity, save_dir="sensitivity_line",
                               ylim=nothing, ci_const = 1.96, kwargs...)

    gr()

    if exp_loc[end] == '/'
        exp_loc = exp_loc[1:end-1]
    end
    head_dir = dirname(exp_loc)
    
    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)
    args = Iterators.product([diff_dict[arg] for arg in product_args]...)

    p1 = ProgressMeter.Progress(length(args), 0.1, "Args: ", offset=0)

    for arg in args

        plt=nothing

        p2 = ProgressMeter.Progress(length(diff_dict[line_arg]), 0.1, "$(line_arg): ", offset=1)

        for (idx_line, l_a) in enumerate(diff_dict[line_arg])

            μ = zeros(length(diff_dict[sweep_arg]))
            σ = zeros(length(diff_dict[sweep_arg]))

            p3 = ProgressMeter.Progress(length(diff_dict[sweep_arg]), 0.1, "$(sweep_arg): ", offset=2)
            for (idx, s_a) in enumerate(diff_dict[sweep_arg])
                search_dict = Dict(sweep_arg=>s_a, line_arg=>l_a, [product_args[idx]=>key for (idx, key) in enumerate(arg)]...)
                _, hashes, _ = search(ic, search_dict)
                μ_runs = zeros(length(hashes))
                for (idx_d, d) in enumerate(hashes)
                    if isfile(joinpath(head_dir, d, results_file))
                        results = load(joinpath(head_dir, d, results_file))
                        μ_runs[idx_d] = clean_func(results)
                        # catch e
                    else
                        # println(joinpath(head_dir, d, results_file))
                        μ_runs[idx_d] = Inf
                    end
                end
                μ[idx] = mean(μ_runs)
                σ[idx] = ci_const * std(μ_runs)/sqrt(length(μ_runs))
                next!(p3)
            end

            if plt == nothing
                plt = plot(sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ, ylim=ylim, label="$(line_arg)=$(l_a)"; kwargs...)
            else
                plot!(plt, sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ, label="$(line_arg)=$(l_a)"; kwargs...)
            end
            next!(p2)
        end

        if !isdir(joinpath(exp_loc, save_dir))
            mkdir(joinpath(exp_loc, save_dir))
        end

        save_file_name = join(["$(key)_$(arg[idx])" for (idx, key) in enumerate(product_args)], "_")

        savefig(plt, joinpath(exp_loc, save_dir, "$(save_file_name).pdf"))
        next!(p1)
    end


end


function sensitivity_best_arg(exp_loc, sweep_arg::String, best_arg::String, product_args::Vector{String};
                              results_file="results.jld2",
                              clean_func=identity,
                              sweep_args_clean=identity,
                              compare=(new, old)->new<old,
                              save_dir="sensitivity_best",
                              ylim=nothing, ci_const = 1.96, kwargs...)

    gr()

    if exp_loc[end] == '/'
        exp_loc = exp_loc[1:end-1]
    end
    head_dir = dirname(exp_loc)
    
    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)
    args = Iterators.product([diff_dict[arg] for arg in product_args]...)

    p1 = ProgressMeter.Progress(length(args), 0.1, "Args: ", offset=0)

    for arg in args

        plt=nothing

        p2 = ProgressMeter.Progress(length(diff_dict[best_arg]), 0.1, "$(best_arg): ", offset=1)

        μ = zeros(length(diff_dict[sweep_arg]))
        fill!(μ, Inf)
        σ = zeros(length(diff_dict[sweep_arg]))

        for (idx_line, b_a) in enumerate(diff_dict[best_arg])

            p3 = ProgressMeter.Progress(length(diff_dict[sweep_arg]), 0.1, "$(sweep_arg): ", offset=2)
            for (idx, s_a) in enumerate(diff_dict[sweep_arg])
                search_dict = Dict(sweep_arg=>s_a, best_arg=>b_a, [product_args[idx]=>key for (idx, key) in enumerate(arg)]...)
                _, hashes, _ = search(ic, search_dict)
                μ_runs = zeros(length(hashes))
                for (idx_d, d) in enumerate(hashes)
                    if isfile(joinpath(head_dir, d, results_file))
                        results = load(joinpath(head_dir, d, results_file))
                        μ_runs[idx_d] = clean_func(results)
                        # catch e
                    else
                        # println(joinpath(head_dir, d, results_file))
                        μ_runs[idx_d] = Inf
                    end
                end
                if compare(mean(μ_runs), μ[idx])
                    μ[idx] = mean(μ_runs)
                    σ[idx] = ci_const * std(μ_runs)/sqrt(length(μ_runs))
                end
                next!(p3)
            end

            next!(p2)
        end

        if plt == nothing
            plt = plot(sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ, ylim=ylim; kwargs...)
        else
            plot!(plt, sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ; kwargs...)
        end

        if !isdir(joinpath(exp_loc, save_dir))
            mkdir(joinpath(exp_loc, save_dir))
        end

        save_file_name = join(["$(key)_$(arg[idx])" for (idx, key) in enumerate(product_args)], "_")

        savefig(plt, joinpath(exp_loc, save_dir, "$(save_file_name).pdf"))
        next!(p1)
    end


end


