import Flux

import Random
import DataStructures

import JuliaRL

# function get_action(env_state, rng=Random.GLOBAL_RNG)
#     rn = rand(rng)
#     if rn < 0.5
#         return 1, 0.5
#     else
#         return 2, 0.5
#     end
# end


mutable struct RGTDAgent{O, GVFN<:GradientGVFN, F, H, Φ, Π, M, G} <: JuliaRL.AbstractAgent
    lu::RGTD
    opt::O
    gvfn::GVFN
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
    prev_action_or_not::Bool
end


function RGTDAgent(horde, out_horde,
                   feature_creator,
                   feature_size,
                   acting_policy::Π,
                   parsed;
                   rng=Random.GLOBAL_RNG,
                   init_func=(dims...)->glorot_uniform(dims...)) where {Π<:AbstractActingPolicy}

    # horde = RingWorldUtils.get_horde(parsed)
    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    lu = RGTD(Float64.(parsed["params"])...)
    τ=parsed["truncation"]

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)

    act = FluxUtils.get_activation(parsed["act"])

    prev_action_or_not = get(parsed, "prev_action_or_not", false)
    
    gvfn = GradientGVFN(feature_size, horde, act; initθ=init_func)

    num_out_gvfs = length(out_horde)
    model = Linear(num_gvfs, num_out_gvfs; init=init_func)

    state_list = DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    hidden_state_init = zeros(Float32, num_gvfs)

    RGTDAgent(lu, opt, gvfn,
              feature_creator,
              state_list,
              hidden_state_init,
              zeros(Float32, 1),
              acting_policy,
              -1, 0.0, model,
              out_horde, zeros(length(horde)),
              prev_action_or_not)

end

function JuliaRL.start!(agent::RGTDAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)

    agent.action, agent.action_prob = agent.π(env_s_tp1, rng)

    fill!(agent.state_list, zeros(length(agent.build_features(env_s_tp1, agent.action))))
    push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t = copy(env_s_tp1)
    return agent.action
end

function JuliaRL.step!(agent::RGTDAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    # Decide Action
    new_action, new_prob = agent.π(env_s_tp1, rng)

    if agent.prev_action_or_not
        push!(agent.state_list, agent.build_features(env_s_tp1, agent.action))
    else
        push!(agent.state_list, agent.build_features(env_s_tp1, new_action))
    end

    update_full_hessian!(agent.gvfn,
            agent.opt,
            agent.lu,
            agent.hidden_state_init,
            agent.state_list,
            env_s_tp1,
            agent.action,
            agent.action_prob)

    preds = GVFN.roll(agent.gvfn,
                      agent.state_list,
                      agent.hidden_state_init,
                      GVFN.Prediction)
    
    agent.preds_tp1 .= preds[end]
    update!(agent.model,
            agent.out_horde,
            agent.opt,
            agent.lu,
            Flux.data.(preds),
            env_s_tp1,
            agent.action,
            agent.action_prob)

    out_preds = agent.model(preds[end])

    agent.action = copy(new_action)
    agent.action_prob = new_prob
    
    agent.s_t .= env_s_tp1
    agent.hidden_state_init .= Flux.data(preds[1])

    return Flux.data.(out_preds), agent.action, preds[end]
end

JuliaRL.get_action(agent::RGTDAgent, state) = agent.action


function Base.print(io::IO, agent::RGTDAgent)
    print(io, Base.string(agent.preds_tp1))
end
