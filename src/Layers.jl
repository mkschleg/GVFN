


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

# sigmoid = Flux.sigmoid
sigmoid′(x) = sigmoid(x)*(1.0-sigmoid(x))
# sigmoid′(x) = begin; tmp = sigmoid(x); tmp*(1.0-tmp); end;



# Linear(in::Integer, out::Integer; init=(dims...)->zeros(Float32, dims...)) =
#     Linear(init(out, in), init(out))

# (layer::Linear)(x) = layer.W*x .+ layer.b
# # prime()

# mutable struct Linear{A, V}
#     W::A
#     b::V
# end

# Linear(in::Integer, out::Integer; init=(dims...)->zeros(Float32, dims...)) =
#     Linear(init(out, in), init(out))

# (layer::Linear)(x) = layer.W*x .+ layer.b





