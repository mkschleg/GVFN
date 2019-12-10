using Reproduce
using Logging
using DataStructures
using NaNMath

include("plot_reproduce.jl")


compassworld_data_clean_func(di) = mean((di["results"]["rmse"]))
compassworld_data_clean_func_end(di, range) = mean((di["results"]["rmse"][range]))
compassworld_data_clean_func_end(di) = compassworld_data_clean_func_end(di, 750000:1000000)

function main(args::Vector{String})
    as = ArgParseSettings(exc_handler=Reproduce.ArgParse.debug_handler)
    @add_arg_table as begin
        "exp_loc"
        arg_type=String
        "--lambda"
        action=:store_true
        "--rnn"
        action=:store_true
        "--actionrnn"
        action=:store_true
        "--extra"
        arg_type=String
        nargs='*'
        "--min"
        arg_type=Int64
        default=750000
        "--max"
        arg_type=Int64
        default=1000000
    end
    parsed = parse_args(args, as)
    exp_loc = parsed["exp_loc"]
    func_dict = Dict(
        "all"=>compassworld_data_clean_func,
        "end"=>(dat)->compassworld_data_clean_func_end(dat, parsed["min"]:parsed["max"]))


    trunc_or_lambda = parsed["lambda"] ? "params" : "truncation"
    horde_or_cell = parsed["rnn"] ? "cell" : "horde"
    

    @info "Plot learning rate sensitivity"

    product_args = [horde_or_cell, trunc_or_lambda, parsed["extra"]...]
    if parsed["actionrnn"]
        product_args = [trunc_or_lambda, parsed["extra"]...]
    end

    sensitivity(exp_loc, "optparams", product_args; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=func_dict["end"], save_dir="sensitivity_alpha_end")
    sensitivity(exp_loc, "optparams", product_args; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=func_dict["all"], save_dir="sensitivity_alpha")

    @info "Plot multiline learning rate sensitivity"

    product_args = [horde_or_cell, parsed["extra"]...]
    if parsed["actionrnn"]
        product_args = [parsed["extra"]...]
    end
    
    sensitivity_multiline(exp_loc, "optparams", trunc_or_lambda, product_args; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=func_dict["all"], save_dir="sensitivity_alpha_multiline")
    
    sensitivity_multiline(exp_loc, "optparams", trunc_or_lambda, product_args; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=func_dict["end"], save_dir="sensitivity_alpha_multiline_end")

    @info "Plot truncation sensitivity"
    
    sensitivity_best_arg(exp_loc, trunc_or_lambda, "optparams", product_args; ylim=(0.0,1.0), clean_func=func_dict["all"], save_dir="sensitivity_trunc")
    sensitivity_best_arg(exp_loc, trunc_or_lambda, "optparams", product_args; ylim=(0.0,1.0), clean_func=func_dict["end"], save_dir="sensitivity_trunc_end")

end

runs_func(μ::Array{<:AbstractFloat}) = Dict(
    "mean"=>mean(μ),
    "stderr"=>std(μ)/length(μ),
    "median"=>length(μ) == 0 ? NaN : median(μ),
    "best"=>NaNMath.minimum(μ),
    "worst"=>NaNMath.maximum(μ))


function synopsis(exp_loc::String, best_args, range)
    
    # Iterators.product
    args = Iterators.product(["mean", "median", "best"], ["all", "end"])
    func_dict = Dict(
        "all"=>compassworld_data_clean_func,
        "end"=>(dat)->compassworld_data_clean_func_end(dat, range))

    if !isdir(joinpath(exp_loc, "synopsis"))
        mkdir(joinpath(exp_loc, "synopsis"))
    end
    
    for a in args
        @info "Current Arg $(a)"
        order_settings(
            exp_loc;
            run_key="seed",
            clean_func=func_dict[a[2]],
            runs_func=runs_func,
            sort_idx=a[1],
            save_locs=[joinpath(exp_loc, "synopsis/order_settings_$(a[1])_$(a[2]).$(ext)") for ext in ["jld2", "txt"]])
    end

    # best_args_str = join(best_args, '_')

    ret = best_settings(exp_loc, best_args;
                        run_key="seed", clean_func=compassworld_data_clean_func,
                        runs_func=runs_func,
                        sort_idx="mean",
                        save_locs=[joinpath(exp_loc, "synopsis/best_$(join(best_args, '_')).txt")])

    ret = best_settings(exp_loc, best_args;
                        run_key="seed", clean_func=func_dict["end"],
                        runs_func=runs_func,
                        sort_idx="mean",
                        save_locs=[joinpath(exp_loc, "synopsis/best_$(join(best_args, '_'))_end.txt")])

end
