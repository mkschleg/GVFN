
using Lazy

abstract type AbstractGVFLayer end

function get(gvfn::AbstractGVFLayer, state_tp1, preds_tp1) end

mutable struct GVFRLayer{F, A, V, T<:AbstractGVF} <: AbstractGVFLayer
    σ::F
    Wx::A
    Wh::A
    b::V
    h::V
    horde::Horde{T}
end

GVFRLayer(num_gvfs, num_ext_features, horde; init=Flux.glorot_uniform, σ_int=σ) =
    GVFRLayer(
        σ_int,
        param(init(num_gvfs, num_ext_features)),
        param(init(num_gvfs, num_gvfs)),
        param(Flux.zeros(num_gvfs)),
        # cumulants,
        # discounts,
        param(Flux.zeros(num_gvfs)),
        horde)

@forward GVFRLayer.horde get
get_question_parameters = get

function (m::GVFRLayer)(h, x)
    new_h = m.σ.(m.Wx*x .+ m.Wh*h)
    return new_h, new_h
end


Flux.hidden(m::GVFRLayer) = m.h
Flux.@treelike GVFRLayer
GVFNetwork(args...; kwargs...) = Flux.Recur(GVFRLayer(args...; kwargs...))

function reset!(m, h_init)
    Flux.reset!(m)
    # println("Hidden state: ", m.state, " ", h_init)
    m.state.data .= Flux.data(h_init)
end

function reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
    Flux.reset!(m)
    # println(h_init)
    m.state[1].data .= Flux.data(h_init[1])
    m.state[2].data .= Flux.data(h_init[2])
end


mutable struct GVFRActionLayer{F, A, B, V, T<:AbstractGVF} <: AbstractGVFLayer
    σ::F
    Wx::A
    Wh::A
    b::B
    h::V
    horde::Horde{T}
end

GVFRActionLayer(num_gvfs, num_actions, num_ext_features, horde; init=Flux.glorot_uniform, σ_int=σ) =
    GVFRActionLayer(
        σ_int,
        param(init(num_actions, num_gvfs, num_ext_features)),
        param(init(num_actions, num_gvfs, num_gvfs)),
        param(Flux.zeros(num_actions, num_gvfs)),
        # cumulants,
        # discounts,
        param(Flux.zeros(num_gvfs)),
        horde)

@forward GVFRActionLayer.horde get

function (m::GVFRActionLayer)(h, x::Tuple{Int64, Array{<:AbstractFloat, 1}})
    new_h = m.σ.(m.Wx[x[1], :, :]*x[2] + m.Wh[x[1], :, :]*h)
    return new_h, new_h
end

Flux.hidden(m::GVFRActionLayer) = m.h
Flux.@treelike GVFRActionLayer
GVFActionNetwork(args...; kwargs...) = Flux.Recur(GVFRActionLayer(args...; kwargs...))


