
mutable struct SingleLayer{F, FP, A, V}
    σ::F
    σ′::FP
    W::A
    b::V
end

SingleLayer(in::Integer, out::Integer, σ, σ′; init=(dims...)->zeros(Float32, dims...)) =
    SingleLayer(σ, σ′, init(out, in), init(out))

(layer::SingleLayer)(x) = layer.σ.(layer.W*x .+ layer.b)
deriv(layer::SingleLayer, x) = layer.σ′.(layer.W*x .+ layer.b)

Linear(in::Integer, out::Integer; kwargs...) =
    SingleLayer(in, out, identity, (x)->1.0; kwargs...)


(l::Flux.Dense)(x::T) where {T<:Tuple} = (x[2], l(x[1]))



mutable struct TCLayer{F, A, V}
    σ::F
    W::A
    b::V
end

Flux.@treelike TCLayer

TCLayer(in::Integer, out::Integer, σ=identity; init=(dims...)->zeros(Float32, dims...)) =
    TCLayer(σ, param(init(out, in)), param(init(out)))

(layer::TCLayer)(x::AbstractArray{Int, 1}) =
    layer.σ(sum(layer.W[:, x];dims=2)[:, 1] .+ layer.b)

(layer::TCLayer)(x::AbstractArray{Int, 2}) =
    layer.σ(sum(layer.W[:, x];dims=2)[:, 1, :] .+ layer.b)
