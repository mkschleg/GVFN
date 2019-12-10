# This is the implementation of the combined gvfn and rnn.

mutable struct GVFRNNCell{F, A, V, H<:AbstractHorde}
    σ::F
    Wx_gvf::A
    Wh_gvf::A
    b_gvfn::V
    h_gvfn::V
    horde::H
    
    Wx_rnn::A
    Wh_rnn::A
    b_rnn::V
    h_rnn::V
end


function (m::GVFRNNCell)(h, x)
    new_h = m.σ.(m.Wx*x .+ m.Wh*h + b_gvfn)
    return new_h, new_h
end


Flux.hidden(m::GVFRNNLayer) = m.h
Flux.@treelike GVFRNNCell
GVFNetwork(args...; kwargs...) = Flux.Recur(GVFRNNLayer(args...; kwargs...))




