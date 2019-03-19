
using GVFN: CycleWorld, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
using ProgressMeter

build_features(s) = [1.0, s[1], 1-s[1]]

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))


function test_GVFN_tdlambda()

    env = CycleWorld(6)
    num_gvfs = 7
    num_steps = 500000
    τ=1

    rng = Random.MersenneTwister(10)

    opt = TDLambda(0.6, 0.9)

    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1), ConstantDiscount(0.0), NullPolicy()) for i in 2:6];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(0.9, ((env_state)->env_state[1] == 1)), NullPolicy())]]


    pred_strg = zeros(num_steps, num_gvfs)

    gvfn = GVFNetwork(num_gvfs, 3, Horde(gvfs); init=(dims...)->glorot_uniform(rng, dims...))
    _, s_t = start!(env)
    h_t = zeros(num_gvfs)
    h_tm1 = zeros(num_gvfs)

    state_list = [zeros(3) for t in 1:τ]
    popfirst!(state_list)
    push!(state_list, build_features(s_t))
    hidden_state_init = zeros(num_gvfs)

    @showprogress 0.1 "Step: " for step in 1:num_steps
        _, s_tp1, _, _ = step!(env, 1)

        if length(state_list) == (τ+1)
            # println(state_list)
            popfirst!(state_list)
        end
        push!(state_list, build_features(s_tp1))

        # print(step, "\r")

        train!(gvfn, opt, hidden_state_init, state_list, s_tp1)

        reset!(gvfn, hidden_state_init)
        preds = gvfn.(state_list)
        pred_strg[step,:] .= preds[end].data
        # println(env.agent_state)
        # println(preds[end])
        s_t .= s_tp1
        hidden_state_init .= Flux.data(preds[1])
        # println(preds)
        # println(hidden_state_init)

    end
    return pred_strg

end
