using Distributed
using Random
using ProgressMeter
using Logging
# using ArgParse


struct ArgIterator
    dict::Dict
    stable_arg::Vector{String}
    arg_list::Vector{String}
    done::Bool
    make_args
    ArgIterator(dict, stable_arg; arg_list=nothing, make_args=nothing) = new(dict, stable_arg, arg_list==nothing ? collect(keys(dict)) : arg_list, false, make_args)
end

function make_arguments(iter::ArgIterator, state)
    arg_list = Vector{String}()

    if iter.make_args == nothing
        new_ret_list = Vector{String}()
        for (arg_idx, arg) in enumerate(iter.arg_list)
            push!(new_ret_list, arg)
            push!(new_ret_list, string(iter.dict[arg][state[2][arg_idx]]))
        end
        arg_list = [new_ret_list; iter.stable_arg]
    else
        d = Dict{String, String}()
        for (arg_idx, arg) in enumerate(iter.arg_list)
            d[arg] = string(iter.dict[arg][state[2][arg_idx]])
        end
        arg_list = [iter.make_args(d); iter.stable_arg]
    end
end

function Base.iterate(iter::ArgIterator)
    state = (1, ones(Int64, length(iter.arg_list)))
    arg_list = make_arguments(iter, state)
    return (state[1], arg_list), next_state(iter, state)
end

function next_state(iter::ArgIterator, _state)
    state = _state[2]
    n_state = _state[1]

    state[end] += 1

    for (arg_idx, arg) in Iterators.reverse(enumerate(iter.arg_list))
        if arg_idx == 1
            return (n_state+1, state)
        end

        if state[arg_idx] > length(iter.dict[arg])
            state[arg_idx] = 1
            state[arg_idx - 1] += 1
        end

    end
end

function Base.iterate(iter::ArgIterator, state)
    if state[2][1] > length(iter.dict[iter.arg_list[1]])
        return nothing
    end
    arg_list = make_arguments(iter, state)
    return (state[1], arg_list), next_state(iter, state)
end

function Base.length(iter::ArgIterator)
    return *([length(iter.dict[key]) for key in iter.arg_list]...)
end

const IN_SLURM = "SLURM_JOBID" in keys(ENV)
IN_SLURM && using ClusterManagers

function parallel_experiment_args(experiment_file, args_iter; exp_module_name=:Main, exp_func_name=:main_experiment, num_workers=5)


    pids = Array{Int64, 1}
    if IN_SLURM
        pids = addprocs(SlurmManager(parse(Int, ENV["SLURM_NTASKS"])))
        print("\n")
    else
        println(num_workers, " ", nworkers())
        if nworkers() == 1
            pids = addprocs(num_workers;exeflags="--project=.")
        elseif nworkers() < num_workers
            pids = addprocs((num_workers) - nworkers();exeflags="--project=.")
        else
            pids = procs()
        end
    end

    println(nworkers(), " ", pids)

    try

        p = Progress(length(args_iter))
        channel = RemoteChannel(()->Channel{Bool}(length(args_iter)), 1)

        mod_str = string(exp_module_name)
        func_str = string(exp_func_name)
        @everywhere global exp_file=$experiment_file
        @everywhere begin
            try
                id = myid()
            catch
                @info "myid not defined?"
                id = 1
            end
        end
        # @everywhere global exp_mod_name=$str
        # @everywhere global exp_f_name=$exp_func_name
        @everywhere begin
            include(exp_file)
            @info "$(exp_file) included on process $(id)"
            exp_func = getfield(getfield(Main, Symbol($mod_str)), Symbol($func_str))
            experiment(args) = exp_func(args)
            @info "Experiment built on process $(id)"
        end

        n = length(args_iter)
        println(n)

        @sync begin
            @async while take!(channel)
                ProgressMeter.next!(p)
            end

            @async begin
                @distributed (+) for (args_idx, args) in collect(args_iter)
                    experiment(args)
                    sleep(0.01)
                    put!(channel,true)
                    0
                end
                put!(channel, false)
            end
        end

    catch ex
        println(ex)
        Distributed.interrupt()
    end

end


function parallel_experiment_test(num_iter; num_workers=5)



    try

        # @everywhere num_iter=num_iter
        p = Progress(num_iter)
        channel = RemoteChannel(()->Channel{Bool}(num_iter), 1)

        # @everywhere global exp_file=$experiment_file
        # @everywhere begin
        #     include(exp_file)
        # end
        # @everywhere exp_func = getfield(getfield(Main, Symbol(exp_module_name)), Symbol(exp_func_name))
        # @everywhere experiment(args) = exp_func(args)
        
        @everywhere function experiment(args, idx)
            sleep(0.1)
            args += 1
        end

        @sync begin
            @async while take!(channel)
                # println("Here")
                ProgressMeter.next!(p)
            end

            @async begin
                @distributed (+) for i in 1:num_iter
                    # println("Here")
                    experiment(i, i)
                    put!(channel, true)
                    # i^2
                    0
                end
                put!(channel, false)
            end
        end

        # @showprogress 0.1 "Experiments: " for (args_idx, args) in args_iter
        #     wait(futures[args_idx])
        # end
    catch ex
        println(ex)
        Distributed.interrupt()
    end

end

