
"""
    ABCEnv

Types of trials:
    ÃB̃
    ÃB
    AB̃
    ABC
"""


module ABCEnvConst

using Random

const N = [0,0,0]
const A = [1,0,0]
const B = [0,1,0]
const C = [0,0,1]

abstract type AbstractTrial end
struct ÃB̃ <: AbstractTrial
    length::Int64
end
# ÃB̃(length, rng::Random.AbstractRNG) = ÃB̃(length)

struct ÃB <: AbstractTrial
    length::Int64
    time_b::Int64
end
# ÃB(length, rng::Random.AbstractRNG) = ÃB(length, random(rng, :length))

struct AB̃ <: AbstractTrial
    length::Int64
    time_a::Int64
end
# AB̃(length, rng::Random.AbstractRNG) = AB̃(length, random(rng, Int64(floor(length/3)):length))

struct ABC <: AbstractTrial
    length::Int64
    time_a::Int64
    time_b::Int64
    time_c::Int64
end
# ABC(length, rng::Random.AbstractRNG) = ABC(length, random())

end


mutable struct TrialABCEnv
    trial::ABCEnvConst.AbstractTrial
    cur_step_in_trial::Int64
end

TrialABCEnv() = TrialABCEnv(ÃB̃(0), 1)

function get_new_trial(rng::Random.AbstractRNG)
    cnst = ABCEnvConst
    rand(rng,
         [cnst.ÃB̃(20),
          cnst.AB̃(20, 5),
          cnst.ÃB(20, 10),
          cnst.ABC(20, 5, 10, 15)])
end

function JuliaRL.reset!(env::TrialABCEnv;
                        rng = Random.GLOBAL_RNG, kwargs...)

    env.trial = get_new_trial(rng)
    env.cur_step_in_trial = 1
end

JuliaRL.get_actions(env::TrialABCEnv) = Set()

function JuliaRL.environment_step!(env::TrialABCEnv,
                                   action;
                                   rng = Random.GLOBAL_RNG,
                                   kwargs...)
    env.cur_step_in_trial 0= 1
end


function JuliaRL.get_reward(env::TrialABCEnv) # -> get the reward of the environment
    return 0
end

get_state(trial::ABCEnvConst.ÃB̃, step) = ABCEnvConst.N
get_state(trial::ABCEnvConst.ÃB, step) = step == trial.time_b ? ABCEnvConst.B : ABCEnvConst.N
get_state(trial::ABCEnvConst.AB̃, step) = step == trial.time_a ? ABCEnvConst.A : ABCEnvConst.N
function get_state(trial::ABCEnvConst.ABC, step)
    if step == trial.time_a
        return ABCEnvConst.A
    elseif step == trial.time_b
        return ABCEnvConst.B
    elseif step == trial.time_c
        return ABCEnvConst.C
    else
        return ABCEnvConst.N
    end
end

function JuliaRL.get_state(env::TrialABCEnv) # -> get state of agent
    get_state(env.trial, env.cur_step_in_trial)
end

JuliaRL.is_terminal(env::TrialABCEnv) =
    env.trial.length == env.cur_step_in_trial

function env_settings!(as::Reproduce.ArgParseSettings,
                       env_type::Type{TrialABCEnv})

end


mutable struct ContABCEnv
    trial::ABCEnvConst.AbstractTrial
    cur_step_in_trial::Int64
end

ContABCEnv() = ContABCEnv(ÃB̃(0), 1)

function get_new_trial(rng::Random.AbstractRNG)
    cnst = ABCEnvConst
    rand(rng,
         [cnst.ÃB̃(20),
          cnst.AB̃(20, 5),
          cnst.ÃB(20, 10),
          cnst.ABC(20, 5, 10, 15)])
end

function JuliaRL.reset!(env::ContABCEnv;
                        rng = Random.GLOBAL_RNG, kwargs...)

    env.trial = get_new_trial(rng)
    env.cur_step_in_trial = 1
end

JuliaRL.get_actions(env::ContABCEnv) = Set()

function JuliaRL.environment_step!(env::ContABCEnv,
                                   action;
                                   rng = Random.GLOBAL_RNG,
                                   kwargs...)

    env.cur_step_in_trial += 1

    if env.trial.length < env.cur_step_in_trial
        env.cur_step_in_trial = 1
        env.trial = get_new_trial(rng)
    end
    
end


function JuliaRL.get_reward(env::ContABCEnv) # -> get the reward of the environment
    return 0
end

function JuliaRL.get_state(env::ContABCEnv) # -> get state of agent
    get_state(env.trial, env.cur_step_in_trial)
end

JuliaRL.is_terminal(env::ContABCEnv) = false


function env_settings!(as::Reproduce.ArgParseSettings,
                       env_type::Type{ContABCEnv})

end
