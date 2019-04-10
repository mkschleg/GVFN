
using Plots, FileIO, Statistics, ProgressMeter



function plot_gvfn_descent(network, algorithm, alphas, truncation, num_runs; n=1000, save_file="")
    gr()
    dir = "compassworld_gvfn/$(network)/$(algorithm)"
    plt = nothing
    for α in alphas
        for τ in truncation
            runs = Array{Array{Float64, 1}, 1}()
            for r in 1:num_runs
                file_loc = joinpath(dir, "Descent_alpha_$(α)_truncation_$(τ)/run_$(r).jld2")
                d = load(file_loc)
                push!(runs, mean(reshape(sqrt.(mean(d["out_err_strg"].^2; dims=2)), n, Integer(size(d["out_err_strg"])[1]/n)); dims=1)[1,:])
            end
            if plt == nothing
                plt = plot(mean(runs), label="alpha: $(α), truncations: $(τ)")
            else
                plot!(mean(runs), lab="alpha: $(α), truncations: $(τ)")
            end
        end
        if save_file==""
            savefig(plt, "$(network)_$(algorithm).pdf")
        else
            savefig(plt, save_file)
        end
    end
end

function plot_gvfn_tdlambda(network, algorithm, alphas, lambdas, num_runs; n=1000, save_file="")
    gr()
    dir = "compassworld_gvfn/$(network)/$(algorithm)"
    plt = nothing
    for α in alphas
        for l in lambdas
            runs = Array{Array{Float64, 1}, 1}()
            for r in 1:num_runs
                file_loc = joinpath(dir, "Descent_alpha_$(α)_lambda_$(l)/run_$(r).jld2")
                d = load(file_loc)
                push!(runs, mean(reshape(sqrt.(mean(d["out_err_strg"].^2; dims=2)), n, Integer(size(d["out_err_strg"])[1]/n)); dims=1)[1,:])
            end
            if plt == nothing
                plt = plot(mean(runs), label="alpha: $(α), lambda: $(l)")
            else
                plot!(mean(runs), lab="alpha: $(α), lambda: $(l)")
            end
        end
        if save_file==""
            savefig(plt, "$(network)_$(algorithm)_$(lambdas).pdf")
        else
            savefig(plt, save_file)
        end
    end
end

function plot_rnn_compassworld(network, cell, alphas, truncation, num_runs; n=1000, optimizer="Descent", load_loc="compassworld_rnn")
    gr()
    dir = "$(load_loc)/$(network)/$(cell)"
    plt = nothing
    for α in alphas
        for τ in truncation
            runs = Array{Array{Float64, 1}, 1}()
            for r in 1:num_runs
                file_loc = joinpath(dir, "$(optimizer)_alpha_$(α)_truncation_$(τ)/run_$(r).jld2")
                d = load(file_loc)
                push!(runs, mean(reshape(sqrt.(mean(d["out_err_strg"].^2; dims=2)), n, Integer(size(d["out_err_strg"])[1]/n)); dims=1)[1,:])
            end
            if plt == nothing
                plt = plot(mean(runs), label="alpha: $(α), truncations: $(τ)")
            else
                plot!(mean(runs), lab="alpha: $(α), truncations: $(τ)")
            end
        end
        savefig(plt, "$(network)_$(cell)_$(optimizer).pdf")
    end
end

function plot_rnn_cycleworld(network, cell, alphas, truncation, num_runs; n=1000)
    gr()
    dir = "cycleworld_rnn/$(network)/$(cell)"
    plt = nothing
    for α in alphas
        for τ in truncation
            runs = Array{Array{Float64, 1}, 1}()
            for r in 1:num_runs
                file_loc = joinpath(dir, "Descent_alpha_$(α)_truncation_$(τ)/run_$(r).jld2")
                d = load(file_loc)
                push!(runs, mean(reshape(mean(d["out_err_strg"].^2; dims=2), n, Integer(size(d["out_err_strg"])[1]/n)); dims=1)[1,:])
            end
            if plt == nothing
                plt = plot(mean(runs), label="alpha: $(α), truncations: $(τ)")
            else
                plot!(mean(runs), lab="alpha: $(α), truncations: $(τ)")
            end
        end
        savefig(plt, "$(network)_$(algorithm).pdf")
    end

end



