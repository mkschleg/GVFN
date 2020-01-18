

using Flux
using Flux.Tracker
using LinearAlgebra
import Reproduce: ArgParseSettings, @add_arg_table

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* Float32(sqrt(2.0f0/sum(dims)))


# Implemented elsewhere
# function reset!(m, h_init)
#     Flux.reset!(m)
#     # println("Hidden state: ", m.state, " ", h_init)
#     m.state.data .= Flux.data(h_init)
# end

# function reset!(m::Flux.Recur{T}, h_init) where {T<:Flux.LSTMCell}
#     Flux.reset!(m)
#     # println(h_init)
#     m.state[1].data .= Flux.data(h_init[1])
#     m.state[2].data .= Flux.data(h_init[2])
# end


function jacobian(δ, pms)
    k  = length(δ)
    J = IdDict()
    for id in pms
        v = get!(J, id, zeros(k, size(id)...))
        for i = 1:k
            Flux.back!(δ[i], once = false) # Populate gradient accumulator
            v[i, :,:] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
    J
end

function jacobian!(J::IdDict, δ::TrackedArray, pms::Params)
    k  = length(δ)
    for i = 1:k
        Flux.back!(δ[i], once = false) # Populate gradient accumulator
        for id in pms
            v = get!(J, id, zeros(typeof(id[1].data), k, size(id)...))::Array{typeof(id[1].data), 3}
            v[i, :, :] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
end

function hessian(δ, pms)

    

    
end

mutable struct StopGradient{T}
    cell::T
    # StopGradient{T}(layer::T) where {T} = new{T}(layer)
end

(layer::StopGradient)(x) = Flux.data(layer.cell(x))

reset!(layer::StopGradient, hidden_state_init) = reset!(layer.cell, hidden_state_init)

function get_clip_coeff(grads,prms; max_norm)
    total_norm=0.0
    for p in prms
        total_norm += norm(Flux.data(grads[p]))^2
    end
    total_norm = sqrt(total_norm)
    clip_coef = max_norm/(total_norm+1e-6)
    return clip_coef < 1.0 ? clip_coef : 1.0
end

function exp_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--exp_loc"
        help="Location of experiment"
        arg_type=String
        default="tmp"
        "--seed"
        help="Seed of rng"
        arg_type=Int64
        default=0
        "--steps"
        help="number of steps"
        arg_type=Int64
        default=100
        "--sweep"
        action=:store_true
        "--prev_action_or_not"
        action=:store_true
        "--verbose"
        action=:store_true
        "--working"
        action=:store_true
        "--progress"
        action=:store_true
    end
end


# Activations
# σ(x::Real) = one(x) / (one(x) + exp(-x))
sigmoid(x) = Flux.σ(x)
function sigmoid′(x)
    s = sigmoid(x)
    s*(1-s)
end
function sigmoid′′(x)
    s = sigmoid(x)
    s*(1-s)*(1-2*s)
end



# Should we export the namespaces? I think not...
include("utils/compassworld.jl")
include("utils/cycleworld.jl")
include("utils/ringworld.jl")
include("utils/flux.jl")
include("utils/timeseries.jl")
include("utils/arg_tables.jl")

include("utils/experiment.jl")
