__precompile__(true)

module CompassWorldTempDependExperiment

using GVFN: CompassWorld, step!, start!
using GVFN
using Statistics
using ProgressMeter
using JLD2
using Reproduce
using Random

function arg_parse(as::ArgParseSettings = ArgParseSettings())

    GVFN.exp_settings!(as)

    #Compass World
    @add_arg_table as begin
        "--policy"
        help="Acting policy of Agent"
        arg_type=String
        default="rafols"
        "--size"
        help="The size of the compass world chain"
        arg_type=Int64
        default=8
    end

    return as
end

function main_experiment(args::Vector{String})

    cwu = GVFN.CompassWorldUtils

    as = arg_parse()
    parsed = parse_args(args, as)

    ######
    # Experiment Setup
    ######
    savepath = ""
    savefile = ""
    if !parsed["working"]
        create_info!(parsed, parsed["exp_loc"]; filter_keys=["verbose", "working", "progress", "exp_loc"])
        savepath = Reproduce.get_save_dir(parsed)
        savefile = joinpath(savepath, "results.jld2")
        if isfile(savefile)
            return
        end
    end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    ######
    # Environment Setup
    ######
    
    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)

    _, s_t = start!(env) # Start environment

    #####
    # Agent specific setup.
    #####
    
    ap = cwu.get_behavior_policy(parsed["policy"])

    action = ap(s_t, rng) # sample action according to policy

    verbose = parsed["verbose"]
    progress = parsed["progress"]

    prg_bar = ProgressMeter.Progress(num_steps, "Step: ")

    num_steps_to_wall = Array{Int64, 1}()
    push!(num_steps_to_wall, 0)

    states_actions = [(copy(s_t), action)]
    
    for step in 1:num_steps

        _, s_tp1, _, _ = GVFN.step!(env, action[1])
        action = ap(s_tp1, rng) # sample action according to policy

        if s_tp1[end] != 1
            push!(num_steps_to_wall, 0)
        else
            num_steps_to_wall[end] += 1
        end
        
        if verbose
            println("step: $(step)")
            println(env)
            println(agent)
            println(out_preds)
        end

        if progress
           next!(prg_bar)
        end

        push!(states_actions, (copy(s_tp1), action))
        
    end

    results = Dict(["num_steps_to_wall"=>num_steps_to_wall, "states_actions"=>states_actions])
    if !parsed["working"]
        JLD2.@save savefile results
    else
        return results
    end
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    main_experiment(ARGS)
    return 0
end

end

# CycleWorldExperiment.main_experiment()


