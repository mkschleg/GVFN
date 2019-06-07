
import JuliaRL

export CycleWorldAgent, CycleWorldRNNAgent

# import CycleWorldUtils
import Flux

import Random
import DataStructures

mutable struct CycleWorldAgent{O, T, F, H, Φ, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    gvfn::Flux.Recur{T}
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    action::Int64
    model::M
    out_horde::Horde{G}
end

build_features_cycleworld(s) = Float32[1.0, s[1], 1-s[1]]

function CycleWorldAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = CycleWorldUtils.get_horde(parsed)

    num_gvfs = length(horde)

    alg_string = parsed["alg"]
    gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    lu = gvfn_lu_func(Float64.(parsed["params"])...)
    τ=parsed["truncation"]

    opt_string = parsed["opt"]
    opt_func = getproperty(Flux, Symbol(opt_string))
    opt = opt_func(Float64.(parsed["optparams"])...)

    act = FluxUtils.get_activation(parsed["act"])

    gvfn = GVFNetwork(num_gvfs, 3, horde; init=(dims...)->0.001*randn(rng, Float32, dims...), σ_int=Flux.σ)
    model = SingleLayer(num_gvfs, 1, σ, sigmoid′)
    out_horde = Horde([GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())])

    state_list = DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    fill!(state_list, zeros(Float32, 3))
    hidden_state_init = zeros(Float32, num_gvfs)

    CycleWorldAgent(lu, opt, gvfn, build_features_cycleworld, state_list, hidden_state_init, zeros(Float32, 3), 1, model, out_horde)

end

function JuliaRL.start!(agent::CycleWorldAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)
    agent.s_t .= env_s_tp1
    fill!(agent.state_list, zeros(3))
    push!(agent.state_list, agent.build_features(env_s_tp1))
    agent.hidden_state_init .= zero(agent.hidden_state_init)
    agent.s_t .= env_s_tp1
    return agent.action
end

function JuliaRL.step!(agent::CycleWorldAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    push!(agent.state_list, agent.build_features(env_s_tp1))

    update!(agent.gvfn, agent.opt, agent.lu, agent.hidden_state_init, agent.state_list, env_s_tp1)

    reset!(agent.gvfn, agent.hidden_state_init)
    preds = agent.gvfn.(agent.state_list)

    update!(agent.model, agent.out_horde, agent.opt, agent.lu, Flux.data.(preds), env_s_tp1)

    out_preds = agent.model(preds[end])

    agent.s_t .= env_s_tp1
    agent.hidden_state_init .= Flux.data(preds[1])
    agent.action = 1

    return Flux.data.(out_preds), agent.action
end

JuliaRL.get_action(agent::CycleWorldAgent, state) = 1


mutable struct CycleWorldRNNAgent{O, T, F, H, Φ, M, G} <: JuliaRL.AbstractAgent
    lu::LearningUpdate
    opt::O
    rnn::Flux.Recur{T}
    out_model::M
    build_features::F
    state_list::DataStructures.CircularBuffer{Φ}
    hidden_state_init::H
    s_t::Φ
    action::Int64
    horde::Horde{G}
end

function CycleWorldRNNAgent(parsed; rng=Random.GLOBAL_RNG)

    horde = CycleWorldUtils.get_horde(parsed)
    num_gvfs = length(horde)

    τ=parsed["truncation"]
    opt = FluxUtils.get_optimizer(parsed)
    rnn = FluxUtils.construct_rnn(3, parsed)
    out_model = Flux.Dense(parsed["numhidden"], length(horde))

    state_list =  DataStructures.CircularBuffer{Array{Float32, 1}}(τ+1)
    hidden_state_init = GVFN.get_initial_hidden_state(rnn)

    CycleWorldRNNAgent(TD(), opt, rnn, out_model, build_features_cycleworld, state_list, hidden_state_init, zeros(Float32, 3), 1, horde)

end

function JuliaRL.start!(agent::CycleWorldRNNAgent, env_s_tp1; rng=Random.GLOBAL_RNG, kwargs...)
    agent.s_t .= env_s_tp1
    fill!(agent.state_list, zeros(3))
    push!(agent.state_list, agent.build_features(env_s_tp1))
    agent.hidden_state_init = FluxUtils.get_initial_hidden_state(agent.rnn)
    agent.s_t .= env_s_tp1
    return agent.action
end

function JuliaRL.step!(agent::CycleWorldRNNAgent, env_s_tp1, r, terminal; rng=Random.GLOBAL_RNG, kwargs...)

    push!(agent.state_list, agent.build_features(env_s_tp1))

    # RNN update function
    update!(agent.out_model, agent.rnn,
            agent.horde, agent.opt,
            agent.lu, agent.hidden_state_init,
            agent.state_list, env_s_tp1)
    # End update function

    Flux.truncate!(agent.rnn)
    reset!(agent.rnn, agent.hidden_state_init)
    rnn_out = agent.rnn.(agent.state_list)
    out_preds = agent.out_model(rnn_out[end])

    agent.hidden_state_init = FluxUtils.get_next_hidden_state(agent.rnn, agent.hidden_state_init, agent.state_list[1])
    agent.s_t .= env_s_tp1
    agent.action = 1

    return Flux.data.(out_preds), agent.action
end

JuliaRL.get_action(agent::CycleWorldRNNAgent, state) = 1
