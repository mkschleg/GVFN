# This is the implementation of the combined gvfn and rnn.

mutable struct GVFRNNCell{F, A, V, H<:AbstractHorde}
    σ::F
    Wx_gvf::A
    Wh_gvf::A
    b_gvfn::V
    h_gvfn::V
    
    Wx_rnn::A
    Wh_rnn::A
    b_rnn::V
    
    h_rnn::V
    horde::H
end


function (m::GVFRNNCell)(h, x)
    new_h = m.σ.(m.Wx*x .+ m.Wh*h + b_gvfn)
    return new_h, new_h
end


Flux.hidden(m::GVFRLayer) = m.h
Flux.@treelike 
GVFNetwork(args...; kwargs...) = Flux.Recur(GVFRLayer(args...; kwargs...))




