import Flux

import Random
using DataStructures: CircularBuffer, isfull



import JuliaRL



mutable struct ForecastAgent{O, T, F, H, Φ, Π, M, G} <: JuliaRL.AbstractAgent
    opt::O
    forecast_rnn::Flux.Recur{T}
    k::Array{Int64, 1}
    k_idx::Array{Int64, 1}
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    τ::Int64
    hidden_state_init::H
    s_t::Φ
    π::Π
    action::Int64
    action_prob::Float64
    model::M
    out_horde::G
    preds_tp1::Array{Float64, 1}
    prev_action_or_not::Bool
end


function ForecastAgent(forecast_obj::Array{Int64, 1},
                       forecast_obj_idx::Array{Int64, 1},
                       out_horde,
                       feature_creator,
                       feature_size,
                       acting_policy::Π,
                       parsed;
                       rng=Random.GLOBAL_RNG,
                       init_func=(dims...)->glorot_uniform(dims...)) where {Π<:AbstractActingPolicy}

    τ=parsed["truncation"]
    
    opt = FluxUtils.get_optimizer(parsed)
    act = FluxUtils.get_activation(parsed["act"])
    prev_action_or_not = get(parsed, "prev_action_or_not", false)

    ####
    # Define Model
    ####
    k_perm = sortperm(forecast_obj)
    if parsed["cell"] != "RNN"
        throw("no other Forecast network type created!")
    end
    forecast_rnn = TargetNetwork(feature_size, length(k_perm), act; init=init_func)
    model = Linear(length(forecast_obj), length(out_horde); init=init_func)

    state_list =
        DataStructures.CircularBuffer{Array{Float32, 1}}(
            τ+forecast_obj[k_perm][end])
    hidden_state_init = zeros(Float32, length(k_perm))

    ForecastAgent(opt,
                  forecast_rnn,
                  forecast_obj[k_perm],
                  forecast_obj_idx[k_perm],
                  feature_creator,
                  state_list,
                  τ,
                  initial_hidden_state(forecast_rnn),
                  zeros(Float32, 1),
                  acting_policy,
                  -1, 0.0,
                  model,
                  out_horde,
                  zeros(length(k_perm)),
                  prev_action_or_not)

end

function agent_settings!(as::Reproduce.ArgParseSettings, agent_type::Type{ForecastAgent})
    Reproduce.@add_arg_table as begin
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
        "--cell"
        help="Cell"
        default="RNN"
        "--act"
        help="Activation"
        default="tanh"
    end

    FluxUtils.opt_settings!(as)
    
end

function JuliaRL.start!(agent::ForecastAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    fill!(agent.state_list,
          zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list,
          agent.build_features(env_s_tp1, agent.action))
    
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end

function JuliaRL.step!(agent::ForecastAgent,
                       env_s_tp1,
                       r,
                       terminal;
                       rng=Random.GLOBAL_RNG,
                       kwargs...)
    
    # Decide Action
    new_action, new_prob = agent.π(env_s_tp1, rng)

    ## NOTE: Fix back to Action_t when done testing.

    if agent.prev_action_or_not
        push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    else
        push!(agent.state_list, agent.build_features(env_s_tp1, new_action))
    end

    out_preds = Flux.zeros(length(agent.out_horde))
    if isfull(agent.state_list)

        targets = getindex.(agent.state_list[agent.τ .+ agent.k], agent.k_idx)
        # println("targets: ", targets)
        
        update!(agent.forecast_rnn,
                agent.opt,
                agent.hidden_state_init,
                view(agent.state_list, 1:agent.τ),
                targets)

        reset!(agent.forecast_rnn, agent.hidden_state_init)
        preds = agent.forecast_rnn.(agent.state_list)
        agent.preds_tp1 .= Flux.data(preds[end])
        update!(agent.model,
                agent.out_horde,
                agent.opt,
                TD(),
                Flux.data.(preds),
                env_s_tp1,
                agent.action,
                agent.action_prob)

        out_preds .= agent.model(agent.preds_tp1)

        agent.hidden_state_init .= Flux.data(preds[1])
    else
        reset!(agent.forecast_rnn, agent.hidden_state_init)
        preds = agent.forecast_rnn.(agent.state_list)
        agent.preds_tp1 .= Flux.data(preds[end])
        out_preds .= agent.model(agent.preds_tp1)

        agent.hidden_state_init .= Flux.data(preds[1])
    end
    
    agent.action = copy(new_action)
    agent.action_prob = new_prob
    
    agent.s_t .= env_s_tp1
    # agent.hidden_state_init .= Flux.data(preds[1])

    return out_preds, agent.action
end

JuliaRL.get_action(agent::ForecastAgent, state) = agent.action


function Base.print(io::IO, agent::ForecastAgent)
    print(io, Base.string(agent.preds_tp1))
end
