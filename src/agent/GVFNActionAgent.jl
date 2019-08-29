import Flux

import Random
import DataStructures

import JuliaRL

mutable struct GVFNActionAgent{O, T, F, H, Φ, Π, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    gvfn::Flux.Recur{T}
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    π::Π
    action::Int64
    action_prob::Float64
    model::M
    out_horde::Horde{G}
    preds_tp1::Array{Float64, 1}
end


function GVFNActionAgent(horde, out_horde,
                         feature_creator,
                         feature_size,
                         num_actions,
                         acting_policy::Π,
                         parsed;
                         rng=Random.GLOBAL_RNG,
                         init_func=(dims...)->glorot_uniform(dims...)) where {Π<:AbstractActingPolicy}

    # horde = RingWorldUtils.get_horde(parsed)
    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func(Float64.(parsed["params"])...)
    τ=parsed["truncation"]

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)

    act = FluxUtils.get_activation(parsed["act"])

    gvfn = GVFActionNetwork(num_gvfs, num_actions, feature_size, horde; init=init_func, σ_int=act)

    num_out_gvfs = length(out_horde)
    model = Linear(num_gvfs, num_out_gvfs; init=init_func)

    state_list = DataStructures.CircularBuffer{Tuple{Int64, Array{Float32, 1}}}(τ+1)
    hidden_state_init = zeros(Float32, num_gvfs)

    GVFNActionAgent(lu, opt, gvfn,
              feature_creator,
              state_list,
              hidden_state_init,
              (1, zeros(Float32, 1)),
              acting_policy,
              -1, 0.0, model,
              out_horde, zeros(length(horde)))

end

function JuliaRL.start!(agent::GVFNActionAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    # agent.action, agent.action_prob = get_action(env_s_tp1, rng)
    # agent.action = sample(rng, agent.π, env_s_tp1)
    # agent.action_prob = get(agent.π, env_s_tp1, agent.action)
    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    fill!(agent.state_list, (1, zeros(length(agent.build_features(env_s_tp1, agent.action)))))
    push!(agent.state_list, (agent.action, agent.build_features(env_s_tp1, agent.action)))
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t = (agent.action, copy(env_s_tp1))
    return agent.action
end

function JuliaRL.step!(agent::GVFNActionAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    # Decide Action
    new_action, new_prob = agent.π(env_s_tp1, rng)
    
    push!(agent.state_list, (agent.action, agent.build_features(env_s_tp1, new_action)))

    update!(agent.gvfn, agent.opt, agent.lu, agent.hidden_state_init, agent.state_list, env_s_tp1, agent.action, agent.action_prob)

    reset!(agent.gvfn, agent.hidden_state_init)
    preds = agent.gvfn.(agent.state_list)
    agent.preds_tp1 .= Flux.data(preds[end])
    update!(agent.model, agent.out_horde, agent.opt, agent.lu, Flux.data.(preds), env_s_tp1, agent.action, agent.action_prob)

    out_preds = agent.model(preds[end])

    agent.action = copy(new_action)
    agent.action_prob = new_prob
    
    agent.s_t = (new_action, copy(env_s_tp1))
    agent.hidden_state_init .= Flux.data(preds[1])

    return Flux.data.(out_preds), agent.action
end

JuliaRL.get_action(agent::GVFNActionAgent, state) = agent.action


function Base.print(io::IO, agent::GVFNActionAgent)
    print(io, Base.string(agent.preds_tp1))
end
