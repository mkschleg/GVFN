


mutable struct ForecastCell{F, A, V} <: AbstractGVFLayer
    σ::F
    Wx::A
    Wh::A
    h::V
    k::Array{Int64}
end


function (m::ForecastCell)(h, x)
    new_h = m.σ.(m.Wx*x + m.Wh*h)
    return new_h, new_h
end


Flux.hidden(m::ForecastCell) = m.h
Flux.@treelike ForecastCell
ForecastNetwork(args...; kwargs...) = Flux.Recur(ForecastNetwork(args...; kwargs...))


function update!(frnn::Flux.Recur{T}, opt, h_init, hist_state_seq, future_state_seq) where {T<:ForecastCell}
    reset!(frnn, h_init)
    ps = Flux.params(frnn)
    targets = future_state_seq[frnn.k]

    # grads = Flux.Tracker.gradient(()->ℒ, Flux.params(frnn))
    gs = gradient(ps) do
        Flux.mse(frnn.(state_seq)[end], targets)
    end
    reset!(frnn, h_init)
    Flux.Tracker.update!(opt, ps, gs)
end



