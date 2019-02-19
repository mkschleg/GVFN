
module SingleLayerGVFN

export GVFNetwork, get_parameters!

mutable struct GVFNetwork
    gvfs::Array{GVF,1}
    mod::Module
end

@forward GVFNetwork.gvfs Base.length

function get_parameters!(gvfn::GVFNetwork, r_tp1, γ_tp1, ρ_tp1, obs_tp1, action_tp1, preds_tp1, behaviour_policy)
    bp_prob = get_probability(behaviour_policy, obs_tp1, action_tp1)
    for (gvf_idx, gvf) in enumerate(gvfn.gvfs)
        r_tp1[gvf_idx] = get_cumulant(gvf, obs_tp1, preds_tp1)
        γ_tp1[gvf_idx] = get_continuation(gvf, obs_tp1, preds_tp1)
        ρ_tp1[gvf_idx] = get_probability(gvf, obs_tp1, action_tp1)/bp_prob
    end
end


module Networks
using LinearAlgebra

mutable struct Network
    weights::Array{Array{Float64, 1}, 1}
    traces::Array{Array{Float64, 1}, 1}
    Network(input_size, num_outputs) = new([zeros(num_inputs) for gvf in 1:num_outputs], [zeros(num_inputs) for output in 1:num_outputs])
end

function optimize_gvfs!(
    network::Network,
    preds::Array{Float64, 1},
    preds_tilde::Array{Float64, 1},
    ϕ_t::Array{Float64, 1},
    ϕ_tp1::Array{Float64, 1},
    r::Array{Float64, 1},
    γ_t::Array{Float64, 1},
    γ_tp1::Array{Float64, 1},
    ρ_t::Array{Float64, 1},
    α::Float64,
    λ::Float64,
    numgvfs::Integer,
    actprime)
    @inbounds for gvf in 1:numgvfs
        δ = r[gvf] + γ_tp1[gvf]*preds_tilde[gvf] - preds[gvf]
        network.traces[gvf] .= ρ_t[gvf].*((γ_t[gvf]*λ).*network.traces[gvf] .+ actprime((network.weights[gvf]'*ϕ_t)).*ϕ_t)
        network.weights[gvf] .+= (α*δ).*network.traces[gvf]
    end
end

function optimize_gvfs_bptt!(
    network::Network,
    preds::Array{Array{Float64, 1}, 1},
    preds_tilde::Array{Array{Float64, 1}},
    ϕ_t::Array{Array{Float64, 1}},
    ϕ_tp1::Array{Array{Float64, 1}},
    r::Array{Array{Float64, 1}},
    γ_t::Array{Array{Float64, 1}},
    γ_tp1::Array{Array{Float64, 1}},
    ρ_t::Array{Array{Float64, 1}},
    α::Float64,
    λ::Float64,
    numgvfs::Integer,
    actprime)

    # Calculate BPTT gradients

    # Update GVFs as above.

end

function make_predictions!(ϕ::Array{Float64, 1}, network::Network, preds::Array{Float64, 1}; activate=Main.sigmoid)
    # preds .= sigmoid.(dot.([ϕ], weights))
    @inbounds for gvf = 1:length(network.weights)
        preds[gvf] = activate(network.weights[gvf]'*ϕ)
    end
end


include("gvfn/CycleWorld.jl")
include("gvfn/CompassWorld.jl")

end
