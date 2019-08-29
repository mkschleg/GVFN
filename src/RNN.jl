
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


mutable struct RNNInvCell{F, A, V}
    σ::F
    Wx::A
    Wh::A
    b::V
    h::V
end

RNNInvCell(in, out, σ=tanh; init=Flux.glorot_uniform) =
    RNNInvCell(
        σ,
        param(init(out, in)),
        param(init(out, out)),
        param(Flux.zeros(out)),
        # cumulants,
        # discounts,
        param(Flux.zeros(out)))

function (m::RNNInvCell)(h, x)
    new_h = m.Wx*x .+ m.Wh*m.σ.(h) .+ m.b
    return new_h, new_h
end

Flux.hidden(m::RNNInvCell) = m.h
Flux.@treelike RNNInvCell
RNNInv(args...; kwargs...) = Flux.Recur(RNNInvCell(args...; kwargs...))


mutable struct RNNInvActionCell{F, A, V, H}
    σ::F
    Wx::A
    Wh::A
    b::V
    h::H
end

RNNInvActionCell(in, out, actions, σ=tanh; init=Flux.glorot_uniform) =
    RNNInvActionCELL(
        σ,
        param(init(actions, out, in)),
        param(init(actions, out, out)),
        param(Flux.zeros(actions, out)),
        # cumulants,
        # discounts,
        param(Flux.zeros(out)))

function (m::RNNInvActionCell)(h, x::Tuple{Int64, Array{<:AbstractFloat, 1}})
    new_h = m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*m.σ.(h) .+ m.b[x[1], :]
    return new_h, new_h
end

Flux.hidden(m::RNNInvActionCell) = m.h
Flux.@treelike RNNInvActionCell
RNNInvAction(args...; kwargs...) = Flux.Recur(RNNInvActionCell(args...; kwargs...))
