
# Sepcifying a action-conditional RNN Cell


mutable struct RNNActionCELL{F, A, V, H}
    σ::F
    Wx::A
    Wh::A
    b::V
    h::H
end

RNNActionCELL(num_hidden, num_actions, num_ext_features; init=Flux.glorot_uniform, σ_int=tanh) =
    RNNActionCELL(
        σ_int,
        param(init(num_actions, num_hidden, num_ext_features)),
        param(init(num_actions, num_hidden, num_hidden)),
        param(zeros(Float32, num_actions, num_hidden)),
        # cumulants,
        # discounts,
        param(Flux.zeros(num_hidden)))

function (m::RNNActionCELL)(h, x::Tuple{Int64, Array{<:AbstractFloat, 1}})
    new_h = m.σ.(m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*h .+ m.b[x[1], :])
    return new_h, new_h
end

Flux.hidden(m::RNNActionCELL) = m.h
Flux.@treelike RNNActionCELL
RNNActionLayer(args...; kwargs...) = Flux.Recur(RNNActionCELL(args...; kwargs...))
