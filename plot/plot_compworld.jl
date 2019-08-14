using Reproduce
using Logging
using DataStructures


include("plot_reproduce.jl")


compassworld_data_clean_func(di) = mean((di["results"]["rmse"]))
compassworld_data_clean_func_end(di) = mean((di["results"]["rmse"][1500000:end]))


function main(args::Vector{String})
    as = ArgParseSettings()
    @add_arg_table as begin
        "exp_loc"
        arg_type=String
        "--lambda"
        action=:store_true
        "--rnn"
        action=:store_true
    end
    parsed = parse_args(args, as)
    exp_loc = parsed["exp_loc"]

    trunc_or_lambda = parsed["lambda"] ? "params" : "truncation"
    horde_or_cell = parsed["rnn"] ? "cell" : "horde"

    @info "Plot learning rate sensitivity"

    sensitivity(exp_loc, "optparams", [horde_or_cell, trunc_or_lambda, "feature"]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=compassworld_data_clean_func_end, save_dir="sensitivity_alp
ha_end")
    sensitivity(exp_loc, "optparams", [horde_or_cell, trunc_or_lambda, "feature"]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=compassworld_data_clean_func, save_dir="sensitivity_alpha")

    @info "Plot multiline learning rate sensitivity"

    sensitivity_multiline(exp_loc, "optparams", trunc_or_lambda, [horde_or_cell, "feature"]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=compassworld_data_clean_func, save_dir="sensitivi
ty_alpha_multiline")
    sensitivity_multiline(exp_loc, "optparams", trunc_or_lambda, [horde_or_cell, "feature"]; sweep_args_clean=(a)->getindex.(a, 1), ylim=(0.0,1.0), clean_func=compassworld_data_clean_func_end, save_dir="sensi
tivity_alpha_multiline_end")

    @info "Plot truncation sensitivity"

    sensitivity_best_arg(exp_loc, trunc_or_lambda, "optparams", [horde_or_cell, "feature"]; ylim=(0.0,1.0), clean_func=compassworld_data_clean_func, save_dir="sensitivity_trunc")
    sensitivity_best_arg(exp_loc, trunc_or_lambda, "optparams", [horde_or_cell, "feature"]; ylim=(0.0,1.0), clean_func=compassworld_data_clean_func_end, save_dir="sensitivity_trunc_end")

end

runs_func(μ::Array{<:AbstractFloat}) = Dict(
    "mean"=>mean(μ),
    "stderr"=>std(μ)/length(μ),
    "median"=>median(μ),
    "best"=>NaNMath.minimum(μ),
    "worst"=>NaNMath.maximum(μ))


function synopsis(exp_loc::String)
    
    # Iterators.product
    args = Iterators.product(["mean", "median", "best"], ["all", "end"])
    func_dict = Dict(
        "all"=>compassworld_data_clean_func,
        "end"=>compassworld_data_clean_func_end)

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
end
