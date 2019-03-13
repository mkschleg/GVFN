

import GVFN: CycleWorld, start!, step!

using Flux
using Flux.Tracker

build_features(s) = [1.0, s[1], 1-s[1]]


sigmoid(x::Float64) = 1.0/(1.0 + exp(-x))
sigmoidprime(x::Float64) = sigmoid(x)*(1.0-sigmoid(x))

# Assuming single layer for now...
mutable struct GVFLayer
    weights::Array{Float64, 2} #
    traces::Array{Float64, 2}
    cumulants::Array{Any, 1}
    discounts::Array{Any, 1}

    GVFLayer(num_gvfs, num_ext_features, cumulants, discounts) =
        new(0.1.*rand(num_gvfs, num_ext_features+num_gvfs), zeros(num_gvfs, num_ext_features+num_gvfs), cumulants, discounts)
end

function (m::GVFLayer)(h, x)
    # println(size([h;x]))
    # println(size(m.weights))
    new_h = sigmoid.(m.weights*[h;x])
    return new_h, new_h
end

function deriv(m::GVFLayer, h, x)
    preds_prime = sigmoidprime.(m.weights*[h;x])
    return preds_prime, preds_prime
end

function simple_train(gvfn::GVFLayer, α, λ, h_tm1, h_t, x_t, x_tp1, env_state_tp1)

    preds_t = h_t

    preds_tilde, preds_tilde = gvfn(h_t, x_tp1)

    cumulants = [gvfn.cumulants[i](env_state_tp1, preds_tilde) for i in 1:length(gvfn.cumulants)]
    discounts = [gvfn.discounts[i](env_state_tp1) for i in 1:length(gvfn.cumulants)]

    targets = cumulants .+ discounts.*gvfn(preds_t, x_tp1)[1]

    δ = targets .- preds_t

    preds_prime = deriv(gvfn, h_tm1, x_t)[1]
    @inbounds for gvf in 1:length(discounts)
        trace_view = view(gvfn.traces, gvf, :)
        trace_view .= 1.0.*(discounts[gvf].*λ.*trace_view .+ preds_prime[gvf].*[h_tm1; x_t])
        gvfn.weights[gvf,:] .+= trace_view.*(α*δ[gvf])
    end
    # gvfn.weights .+= gvfn.traces.*(α.*δ)
end

function test()

    env = CycleWorld(6)

    discount = (args...)->0.0
    cumulant(i) = (s_tp1, p_tp1)-> i==1 ? s_tp1[1] : p_tp1[i-1]
    cumulants = [[cumulant(i) for i in 1:6]; [cumulant(1)]]
    discounts = [[discount for i in 1:6]; [(env_state)-> env_state[1] == 1 ? 0.0 : 0.9]]
    # println(size(discounts))

    gvflayer = GVFLayer(7, 3, cumulants, discounts)
    _, s_t = start!(env)
    h_t = zeros(7)
    h_tm1 = zeros(7)
    for step in 1:500000
        _, s_tp1, _, _ = step!(env, 1)
        println(step)
        print(env)
        h_t, h_t = gvflayer(h_tm1, build_features(s_t))
        # println(h_t)
        # h_tp1, h_tp1 = gvflayer(h_t, build_features(s_tp1))
        simple_train(gvflayer, 0.8, 0.9, h_tm1, h_t, build_features(s_t), build_features(s_tp1), s_tp1)
        h_t, h_t = gvflayer(h_tm1, build_features(s_t))
        println(h_t)

        s_t .= s_tp1
        h_tm1 .= h_t
        # h_t .= h_tp1
    end


end


