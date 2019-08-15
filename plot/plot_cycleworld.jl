using Reproduce
using Logging
using NaNMath

include("../plot/plot_reproduce.jl")

cycleworld_data_clean_func(di) = mean(abs.(di["results"]["out_err_strg"]))
cycleworld_data_clean_func_end(di) = mean(abs.(di["results"]["out_err_strg"][250000:300000]))


function main(args::Vector{String}=ARGS)

    as = ArgParseSettings()
    @add_arg_table as begin
        "exp_loc"
        arg_type=String
    end
    parsed = parse_args(args, as)
    exp_loc = parsed["exp_loc"]

    ic = ItemCollection(exp_loc)
    diff_dict = diff(ic.items)
    
    trunc_or_lambda = "truncation" ∉ keys(diff_dict) ? "params" : "truncation"
    horde_or_cell = "cell" ∈ keys(diff_dict) ? "cell" : "horde"

    @info "Plot learning rate sensitivity"

    sensitivity(exp_loc, "optparams", ["act", horde_or_cell, trunc_or_lambda]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=cycleworld_data_clean_func_end, save_dir="sensitivity_alpha_end")
    sensitivity(exp_loc, "optparams", ["act",horde_or_cell, trunc_or_lambda]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=cycleworld_data_clean_func, save_dir="sensitivity_alpha")

    @info "Plot multiline learning rate sensitivity"

    sensitivity_multiline(exp_loc, "optparams", trunc_or_lambda, ["act", horde_or_cell]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=cycleworld_data_clean_func, save_dir="sensitivity_alpha_multiline")
    sensitivity_multiline(exp_loc, "optparams", trunc_or_lambda, ["act", horde_or_cell]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=cycleworld_data_clean_func_end, save_dir="sensitivity_alpha_multiline_end")

    @info "Plot truncation sensitivity"

    sensitivity_best_arg(exp_loc, trunc_or_lambda, "optparams", ["act", horde_or_cell]; ylim=(0.0,1.0), clean_func=cycleworld_data_clean_func, save_dir="sensitivity_trunc")
    sensitivity_best_arg(exp_loc, trunc_or_lambda, "optparams", ["act", horde_or_cell]; ylim=(0.0,1.0), clean_func=cycleworld_data_clean_func_end, save_dir="sensitivity_trunc_end")

end


runs_func(μ::Array{<:AbstractFloat}) = Dict(
    "mean"=>mean(μ),
    "stderr"=>std(μ)/length(μ),
    "median"=>length(μ) == 0 ? Inf : median(μ),
    "best"=>NaNMath.minimum(μ),
    "worst"=>NaNMath.maximum(μ))


function synopsis(exp_loc::String; best_args=["truncation", "horde"])
    
    # Iterators.product
    args = Iterators.product(["mean", "median", "best"], ["all", "end"])
    func_dict = Dict(
        "all"=>cycleworld_data_clean_func,
        "end"=>cycleworld_data_clean_func_end)

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

    ret = best_settings(exp_loc, best_args;
                        run_key="seed", clean_func=cycleworld_data_clean_func,
                        runs_func=runs_func,
                        sort_idx="mean",
                        save_locs=[joinpath(exp_loc, "best_trunc_horde.txt")])

    ret = best_settings(exp_loc, best_args;
                        run_key="seed", clean_func=cycleworld_data_clean_func_end,
                        runs_func=runs_func,
                        sort_idx="mean",
                        save_locs=[joinpath(exp_loc, "best_trunc_horde_end.txt")])
end
