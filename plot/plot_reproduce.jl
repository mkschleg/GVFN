using Plots
using Reproduce
using Statistics
using ProgressMeter
using FileIO


function sensitivity(exp_loc, sweep_arg::String, product_args::Vector{String};
                     results_file="results.jld2", clean_func=identity,
                     ci_const = 1.96, sweep_args_clean=identity, save_dir="sensitivity", ylim=nothing)

    gr()

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
                try
                    results = load(joinpath(d, results_file))
                    μ_runs[idx_d] = clean_func(results)
                catch e
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
                    try
                        results = load(joinpath(d, results_file))
                        μ_runs[idx_d] = clean_func(results)
                    catch e
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
                    try
                        results = load(joinpath(d, results_file))
                        μ_runs[idx_d] = clean_func(results)
                    catch e
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

function learning_curves(exp_loc, line_arg::String, sweep_arg::String, product_args::Vector{String};
                         results_file="results.jld2",
                         clean_func=identity,
                         sweep_args_clean=identity,
                         compare=(new, old)->new<old,
                         save_dir="sensitivity_best",
                         ylim=nothing, ci_const = 1.96, n=1, kwargs...)


    gr()

    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)
    args = Iterators.product([diff_dict[arg] for arg in product_args]...)

    p1 = ProgressMeter.Progress(length(args), 0.1, "Args: ", offset=0)

    p2 = ProgressMeter.Progress(length(diff_dict[line_arg]), 0.1, "$(line_arg): ", offset=1)

    for arg in args

        plt=nothing
        for (idx_line, l_a) in enumerate(diff_dict[line_arg])

            μ_dict = Dict{Any, Any}() #zeros(length(diff_dict[sweep_arg]))

            p3 = ProgressMeter.Progress(length(diff_dict[sweep_arg]), 0.1, "$(sweep_arg): ", offset=2)
            for (idx, s_a) in enumerate(diff_dict[sweep_arg])
                search_dict = Dict(sweep_arg=>s_a, line_arg=>l_a, [product_args[idx]=>key for (idx, key) in enumerate(arg)]...)
                _, hashes, _ = search(ic, search_dict)
                μ_runs = nothing#zeros(length(hashes))
                inf_list = Int64[]
                for (idx_d, d) in enumerate(hashes)
                    try
                        results = load(joinpath(d, results_file))
                        if μ_runs == nothing
                            r = clean_func(results)
                            μ_runs = zeros(length(hashes), length(r))
                            μ_runs[idx_d, :] .= r
                        else
                            μ_runs[idx_d, :] .= r
                        end
                    catch e
                        if μ_runs == nothing
                            push!(inf_list, copy(idx_d))
                        else
                            μ_runs[idx_d, :] = Inf
                        end
                    end
                end
                for idx_inf in inf_list
                    μ_runs[idx_inf, :] = Inf
                end
                # μ[idx] = mean(μ_runs)
                μ_dict[s_a] = mean(μ_runs;dims=1)

                # σ[idx] = ci_const * std(μ_runs)/sqrt(length(μ_runs))
                next!(p3)
            end

            if plt == nothing
                plt = plot(sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ, ylim=ylim, label="$(line_arg)=$(l_a)"; kwargs...)
            else
                plot!(plt, sweep_args_clean(diff_dict[sweep_arg]), μ, yerror=σ, label="$(line_arg)=$(l_a)"; kwargs...)
            end
            next!(p2)
        end
    end
end


