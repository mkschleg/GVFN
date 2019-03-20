

using Flux


mutable struct StopGradient{T}
    cell::T
end

(layer::StopGradient)(x) = Flux.data(layer.cell(x))


