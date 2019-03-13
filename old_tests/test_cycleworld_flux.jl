
import GVFN: CycleWorld, start!, step!

using Flux
using Flux.Tracker
import LinearAlgebra.Diagonal

build_features(s) = [1.0, s[1], 1-s[1]]

# sigmoid(x::Float64) = 1.0/(1.0 + exp(-x))
# sigmoidprime(x::Float64) = sigmoid(x)*(1.0-sigmoid(x))

# Assuming single layer for now...
mutable struct GVFLayer{F, A}
    σ::F
    weights::A
    traces::A
    cumulants::AbstractArray
    discounts::AbstractArray
    # h::V
end

GVFLayer(num_gvfs, num_ext_features, cumulants, discounts; init=Flux.glorot_uniform, σ_int=σ) =
    GVFLayer(
        σ_int,
        param(0.1.*init(num_gvfs, num_ext_features+num_gvfs)),
        param(Flux.zeros(Float64, num_gvfs, num_ext_features+num_gvfs)),
        cumulants,
        discounts)

function (m::GVFLayer)(h, x)
    # println(size([h;x]))
    # println(size(m.weights))
    new_h = m.σ.(m.weights*[h; x])
    return new_h, new_h
end

function simple_train(gvfn::GVFLayer, α, λ, h_tm1, h_t, x_t, x_tp1, env_state_tp1)
    preds_t = h_t
    preds_tilde, preds_tilde = gvfn(h_t, x_tp1)
    cumulants = [gvfn.cumulants[i](env_state_tp1, preds_tilde.data) for i in 1:length(gvfn.cumulants)]
    discounts = [gvfn.discounts[i](env_state_tp1) for i in 1:length(gvfn.cumulants)]
    targets = cumulants .+ discounts.*gvfn(preds_t, x_tp1)[1].data
    δ = targets .- preds_t
    grads = Tracker.gradient(() -> sum(δ), Params([gvfn.weights]))
    gvfn.traces = convert(Array{Float64, 2}, Diagonal(discounts)) * λ * gvfn.traces - grads[gvfn.weights]
    Flux.Tracker.update!(gvfn.weights, α.*gvfn.traces.*(δ))
end

function test()

    env = CycleWorld(6)
    num_gvfs = 7
    num_steps = 500000

    discount = (args...)->0.0
    cumulant(i) = (s_tp1, p_tp1)-> i==1 ? s_tp1[1] : p_tp1[i-1]
    cumulants = [[cumulant(i) for i in 1:6]; [cumulant(1)]]
    discounts = [[discount for i in 1:6]; [(env_state)-> env_state[1] == 1 ? 0.0 : 0.9]]

    pred_strg = zeros(num_steps, num_gvfs)

    gvflayer = GVFLayer(7, 3, cumulants, discounts)
    _, s_t = start!(env)
    h_t = zeros(7)
    h_tm1 = zeros(7)
    for step in 1:num_steps
        _, s_tp1, _, _ = step!(env, 1)
        print(step, "\r")
        # print(env)
        h_t, h_t = gvflayer(h_tm1, build_features(s_t))
        # println(h_t)
        # h_tp1, h_tp1 = gvflayer(h_t, build_features(s_tp1))
        simple_train(gvflayer, 0.8, 0.9, h_tm1, h_t, build_features(s_t), build_features(s_tp1), s_tp1)
        h_t, h_t = gvflayer(h_tm1, build_features(s_t))
        pred_strg[step,:] .= h_t.data
        # println(h_t)

        s_t .= s_tp1
        h_tm1 .= h_t.data
    end
    return pred_strg

end


