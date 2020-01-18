


mutable struct TargetCell{F, A, V} <: AbstractGVFRCell
    σ::F
    Wx::A
    Wh::A
    b::V
    h::V
end

TargetCell(in, out, σ=tanh; init=Flux.glorot_uniform) =
    TargetCell(
        σ,
        param(init(out, in)),
        param(init(out, out)),
        param(Flux.zeros(out)),
        param(Flux.zeros(out)))

function (m::TargetCell)(h, x)
    new_h = m.σ.(m.Wx*x .+ m.Wh*h .+ m.b)
    return new_h, new_h
end


Flux.hidden(m::TargetCell) = m.h
Flux.@treelike TargetCell
TargetNetwork(args...; kwargs...) = Flux.Recur(TargetCell(args...; kwargs...))

initial_hidden_state(m::Flux.Recur{T}) where {T<:TargetCell} = Flux.zeros(length(m.cell.b))

function update!(frnn::Flux.Recur{T}, opt, h_init, hist_state_seq, targets) where {T<:TargetCell}
    reset!(frnn, h_init)
    ps = Flux.params(frnn)
    grads = Flux.Tracker.gradient(ps) do
        Flux.mse(frnn.(hist_state_seq)[end], targets)
    end
    reset!(frnn, h_init)
    Flux.Tracker.update!(opt, ps, grads)
end


mutable struct TargetActionCell{F, A, B, V} <: AbstractGVFRCell
    σ::F
    Wx::A
    Wh::A
    b::B
    h::V
end

TargetActionCell(num_actions, in, out, σ=tanh; init=Flux.glorot_uniform) =
    TargetActionCell(
        σ,
        param(init(num_actions, out, in)),
        param(init(num_actions, out, out)),
        param(Flux.zeros(num_actions, out)),
        param(Flux.zeros(out)))

function (m::TargetActionCell)(h, x::Tuple{Int64, Array{<:Number, 1}})
    new_h = m.σ.(m.Wx[x[1], :, :]*x[2] .+ m.Wh[x[1], :, :]*h .+ m.b[x[1], :])
    return new_h, new_h
end

Flux.hidden(m::TargetActionCell) = m.h
Flux.@treelike TargetActionCell
TargetActionNetwork(args...; kwargs...) = Flux.Recur(TargetActionCell(args...; kwargs...))

initial_hidden_state(m::Flux.Recur{T}) where {T<:TargetActionCell} = Flux.zeros(length(m.cell.h))

function update!(frnn::Flux.Recur{T}, opt, h_init, hist_state_seq, targets) where {T<:TargetActionCell}
    reset!(frnn, h_init)
    ps = Flux.params(frnn)
    grads = Flux.Tracker.gradient(ps) do
        Flux.mse(frnn.(hist_state_seq)[end], targets)
    end
    reset!(frnn, h_init)
    Flux.Tracker.update!(opt, ps, grads)
end

