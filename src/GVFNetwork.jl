
using Lazy

abstract type AbstractGVFLayer end

function get(gvfn::AbstractGVFLayer, state_tp1, preds_tp1) end

mutable struct GVFLayer{F, A, V, T<:AbstractGVF} <: AbstractGVFLayer
    σ::F
    Wx::A
    Wh::A
    h::V
    horde::Horde{T}
end

GVFLayer(num_gvfs, num_ext_features, horde; init=Flux.glorot_uniform, σ_int=σ) =
    GVFLayer(
        σ_int,
        param(0.1.*init(num_gvfs, num_ext_features)),
        param(0.1.*init(num_gvfs, num_gvfs)),
        # cumulants,
        # discounts,
        param(Flux.zeros(num_gvfs)),
        horde)

@forward GVFLayer.horde get
get_question_parameters = get

function (m::GVFLayer)(h, x)
    new_h = m.σ.(m.Wx*x + m.Wh*h)
    return new_h, new_h
end


Flux.hidden(m::GVFLayer) = m.h
Flux.@treelike GVFLayer
GVFNetwork(args...; kwargs...) = Flux.Recur(GVFLayer(args...; kwargs...))

function reset!(m, h_init)
    Flux.reset!(m)
    m.state.data .= h_init
end






