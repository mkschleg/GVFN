
using Plots, FileIO, Statistics, ProgressMeter



function plot_gvfn_descent(network, algorithm, alphas, truncation, num_runs; n=1000)
    gr()
    dir = "compassworld_gvfn/$(network)/$(algorithm)"
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



