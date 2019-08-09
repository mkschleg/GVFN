using Reproduce
using Plots
using FileIO
using ProgressMeter
using Statistics



function collect_data(exp_loc;
                      run_arg="run",
                      results_file="results.jld2", settings_file="settings.jld2", clean_func=identity,
                      save_dir="collected")
                      

    if exp_loc[end] == '/'
        exp_loc = exp_loc[1:end-1]
    end
    head_dir = dirname(exp_loc)

    if !isdir(joinpath(exp_loc, save_dir))
        mkdir(joinpath(exp_loc, save_dir))
    end

    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)
    # args = Iterators.product([diff_dict[arg] for arg in product_args]...)

    search_dict = Dict(run_arg=>diff_dict[run_arg][1])

    _, hashes, _ = search(ic, search_dict)

    settings_vec = Vector{Dict}(undef, length(hashes))

    # collect the parameter settings run
    for (idx, h) in enumerate(hashes)
        sett = load(joinpath(head_dir, h, settings_file))["parsed_args"]
        settings_vec[idx] = Dict(k=>sett[k] for k in filter(v -> v != run_arg, keys(diff_dict)))
    end

    @showprogress for (idx, stngs) in enumerate(settings_vec)
        # println(length(search(ic, stngs)[2]))
        hashes = search(ic, stngs)[2]

        v = Vector{Any}(undef, length(hashes))
        for (idx, h) in enumerate(hashes)
            v[idx] = clean_func(load(joinpath(head_dir, h, results_file)))
        end
        save(joinpath(exp_loc, save_dir, join(["$(k)_$(stngs[k])" for k in keys(stngs)], '_')*".jld2"), Dict("results"=>v, "settings"=>stngs))
    end

end

function collect_sens_data(exp_loc, sens_param, product_args;
                           run_arg="run",
                           results_file="results.jld2", settings_file="settings.jld2", clean_func=identity,
                           save_dir="collected_sens")

    if exp_loc[end] == '/'
        exp_loc = exp_loc[1:end-1]
    end
    head_dir = dirname(exp_loc)

    if !isdir(joinpath(exp_loc, save_dir))
        mkdir(joinpath(exp_loc, save_dir))
    end

    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)
    args = Iterators.product([diff_dict[arg] for arg in product_args]...)

    println(collect(args))

    for arg in collect(args)

        println([k=>arg[k_idx] for (k_idx, k) in enumerate(product_args)])
        
        search_dict = Dict(run_arg=>diff_dict[run_arg][1], [k=>arg[k_idx] for (k_idx, k) in enumerate(product_args)]...)

        _, hashes, _ = search(ic, search_dict)

        settings_vec = Vector{Dict}(undef, length(hashes))

        # collect the parameter settings run
        for (idx, h) in enumerate(hashes)
            sett = load(joinpath(head_dir, h, settings_file))["parsed_args"]
            settings_vec[idx] = Dict([k=>sett[k] for k in filter(v -> v ∉ keys(search_dict), keys(diff_dict))]..., [k=>arg[k_idx] for (k_idx, k) in enumerate(product_args)]...)
        end

        avg_res = zeros(length(diff_dict[sens_param]))
        std_err = zeros(length(diff_dict[sens_param]))
        
        for (idx, stngs) in enumerate(settings_vec)
            # println(length(search(ic, stngs)[2]))
            hashes = search(ic, stngs)[2]
            
            v = zeros(length(hashes))
            for (idx, h) in enumerate(hashes)
                v[idx] = clean_func(load(joinpath(head_dir, h, results_file)))
            end
            println(stngs, ": ", v)

            sens_idx = findfirst(x->x==stngs[sens_param], diff_dict[sens_param])
            avg_res[sens_idx] = mean(v)
            std_err[sens_idx] = std(v)/sqrt(length(hashes))

        end

        save(joinpath(exp_loc, save_dir, "collect_"*join(["$(k)_$(arg[k_idx])" for (k_idx, k) in enumerate(product_args)], '_')*".jld2"), Dict("avg"=>avg_res, "std_err"=>std_err, "sens"=>diff_dict[sens_param], "settings"=>Dict([k=>arg[k_idx] for (k_idx, k) in enumerate(product_args)])))
    end
end


function plot_sens_files(file_list, line_settings_list, save_file="tmp.pdf", ci = 1.97; plot_back=gr, kwargs...)

    plot_back()

    plt = nothing

    for (idx, f) in enumerate(file_list)

        ret = load(f)
        println(ret)

        if plt == nothing
            plt = plot(ret["sens"], ret["avg"], ribbon=ci.*ret["std_err"]; line_settings_list[idx]..., kwargs...)
        else
            plot!(plt, ret["sens"], ret["avg"], ribbon=ci.*ret["std_err"]; line_settings_list[idx]..., kwargs...)
        end
    end

    savefig(plt, save_file)

end

function plot_lc_files(file_list, line_settings_list; save_file="tmp.pdf", ci=1.97, n=1, clean_func=identity, plot_back=gr, kwargs...)

    plot_back()

    plt = nothing

    for (idx, f) in enumerate(file_list)

        ret = load(f)
        l = length(clean_func(ret["results"][1]))
        avg = mean([mean(reshape(clean_func(v), n, Int64(l/n)); dims=1) for v in ret["results"]])'
        std_err = (std([mean(reshape(clean_func(v), n, Int64(l/n)); dims=1) for v in ret["results"]])./sqrt(length(ret["results"])))'

        x = 0:n:l

        if plt == nothing
            plt = plot(avg, ribbon=ci.*std_err; line_settings_list[idx]..., kwargs...)
        else
            plot!(plt, avg, ribbon=ci.*std_err; line_settings_list[idx]..., kwargs...)
        end
    end

    savefig(plt, save_file)
    
end

