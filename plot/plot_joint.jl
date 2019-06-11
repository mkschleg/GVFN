using Plots
using Reproduce
using FileIO
using Statistics
using ProgressMeter


function plot_experiment(line_key, sweep_key; base_dir="cycleworld_joint_sweep", range=1:200000, posix="")
    gr()

    dict = Dict(
        "alpha"=>[0.01],
        "truncation"=>[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "outhorde"=>["onestep", "chain", "gamma_chain"],
        "seed"=>[1, 2, 3, 4, 5],
        "cell"=>["RNN"],
        "beta"=>0.0:0.2:1.0)
	  arg_list = ["outhorde", "cell", "alpha", "beta", "truncation", "seed"]
	  stable_arg = ["--steps", "200000",
                  "--exp_loc", "cycleworld_joint_sweep",
                  "--gvfnhorde", "gamma_chain",
                  "--gvfngamma", "0.9"]

    p1 = ProgressMeter.Progress(length(dict["outhorde"]), 0.1, "Horde: ", offset=0)
    for outhorde in dict["outhorde"] # graph
        p2 = ProgressMeter.Progress(length(dict[line_key]), 0.1, line_key*": ", offset=1)
        plt = nothing
        for lk in dict[line_key] # line
            p3 = ProgressMeter.Progress(length(dict[sweep_key]), 0.1, sweep_key*": ", offset=2)
            tmp_line = zeros(length(dict[sweep_key]))
            tmp_std = zeros(length(dict[sweep_key]))
            for (sk_idx, sk) in enumerate(dict[sweep_key]) # point
                search_dict = Dict(line_key=>lk, sweep_key=>sk, "outhorde"=>outhorde)
                hashes, dirs, items = search(base_dir, search_dict)
                tmp = zeros(length(dirs))
                for (d_idx, d) in enumerate(dirs)
                    tmp_dict = FileIO.load(joinpath(d, "results.jld2"))
                    err = tmp_dict["results"]["out_err"]
                    if size(err)[2] == 1
                        tmp[d_idx] = mean(abs.(err[range, :]))
                    else
                        tmp[d_idx] = mean(sqrt.(mean(err[range, :].^2; dims=2)))
                    end
                end
                tmp_line[sk_idx] = mean(tmp)
                tmp_std[sk_idx] = std(tmp)/sqrt(length(tmp))
                ProgressMeter.next!(p3)
            end
            if plt == nothing
                plt = plot(dict[sweep_key], tmp_line, yerror=tmp_std, label="$(line_key)-$(lk)")
            else
                plot!(dict[sweep_key], tmp_line, yerror=tmp_std, label="$(line_key)-$(lk)")
            end
            ProgressMeter.next!(p2)
        end
        savefig(plt, joinpath(base_dir, "$(outhorde)_$(line_key)$(posix).pdf"))
        ProgressMeter.next!(p1)
    end
end



