
module CycleWorld

export GammaChain, Chain

module GammaChain

using GVFN
using GVFN.Environments
using LinearAlgebra
using Lazy



# const CumulantType = ["Observation", "Composition", "Reward"]
# @enum CumulantType begin
#     Observation = 1
#     Composition = 2
#     Reward = 3
# end

# mutable struct Cumulant
#     typ::CumulantType
#     dependent::Int64
# end

# function get_cumulant(cumulant::Cumulant, obs, preds_tilde)
#     if cumulant.typ == Observation::CumulantType
#         return obs[cumulant.dependent]
#     elseif cumulant.typ == Composition::CumulantType
#         # println("Composition")
#         return preds_tilde[cumulant.dependent]
#     else
#         return 0
#     end
# end

# # const ContinuationType = ["Observation", "Constant"]
# @enum ContinuationType begin
#     Obs = 1
#     Constant = 2
#     Myopic = 3
# end

# mutable struct Continuation
#     typ::ContinuationType
#     γ_const::Float64
#     terminate
# end

# function get_continuation(continuation::Continuation, obs, preds_tilde)
#     if continuation.typ == Obs::ContinuationType
#         if continuation.terminate(obs)
#             return 0.0
#         else
#             return continuation.γ_const
#         end
#     elseif continuation.typ == Constant::ContinuationType
#         return continuation.γ_const
#     else
#         return 0.0
#     end
# end

# mutable struct Policy
#     probabilities::Array{Float64, 1}
# end

# function get_importance_sampling(policy::Policy, action)
#     return 1.0
# end

# mutable struct GVF
#     cumulant::Cumulant
#     continuation::Continuation
#     policy::Policy
# end

# @forward GVF.cumulant get_cumulant
# @forward GVF.continuation get_continuation
# @forward GVF.policy get_importance_sampling

# mutable struct GVFNetwork
#     gvfs::Array{GVF,1}
# end

terminate_on_obs(obs) = obs[2] == 1.0

function build_gvfn()
    gvfn = GVFNetwork([GVF(Cumulant(CumulantTypes.Observation, 2), Continuation(ContinuationTypes.Constant, 0.0, nothing), Policy([1.0])),
                       GVF(Cumulant(CumulantTypes.Composition, 1), Continuation(ContinuationTypes.Constant, 0.0, nothing), Policy([1.0])),
                       GVF(Cumulant(CumulantTypes.Composition, 2), Continuation(ContinuationTypes.Constant, 0.0, nothing), Policy([1.0])),
                       GVF(Cumulant(CumulantTypes.Composition, 3), Continuation(ContinuationTypes.Constant, 0.0, nothing), Policy([1.0])),
                       GVF(Cumulant(CumulantTypes.Composition, 4), Continuation(ContinuationTypes.Constant, 0.0, nothing), Policy([1.0])),
                       GVF(Cumulant(CumulantTypes.Composition, 5), Continuation(ContinuationTypes.Constant, 0.0, nothing), Policy([1.0])),
                       GVF(Cumulant(CumulantTypes.Observation, 2), Continuation(ContinuationTypes.Observation, 0.9, terminate_on_obs), Policy([1.0]))])
    return gvfn
end

# function get_parameters!(gvfn::GVFNetwork, r, γ_t, ρ_t, obs, preds_tilde)
#     for (gvf_idx, gvf) in enumerate(gvfn.gvfs)
#         r[gvf_idx] = get_cumulant(gvf, obs, preds_tilde)
#         γ_t[gvf_idx] = get_continuation(gvf, obs, preds_tilde)
#         ρ_t[gvf_idx] = get_importance_sampling(gvf, action)
#     end
# end


const weight_dim = (7, 10)
const γ_const = 0.9

const ORACLE = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0, γ_const^5], # > GVF 1 prediction when GVF 2
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, γ_const^4],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, γ_const^3],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, γ_const^2],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, γ_const^1],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, γ_const^0]]

function get_parameters!(r, γ_t, ρ_t, obs, preds_tilde)
    r[1] = obs[2]
    for i in 2:(length(preds_tilde) - 1)
        r[i] = preds_tilde[i-1]
    end
    r[end] = obs[2]
    γ_t[end] = 0.9*(1-r[1])
end

@inline sqr_error(env::CycleWorld.Environment, preds::Array{Float64}) = (preds .- ORACLE[env.pos+1]).^2
@inline sqr_error!(env::CycleWorld.Environment, preds::Array{Float64}, loc) = (loc .= (preds .- ORACLE[env.pos+1]).^2)

function get_optimal_weights(bptt)
    if bptt
        return [[-5.96883,  -7.44558,  1.47676,   3.97566,   6.71015,    -4.1983,  -4.52556, -4.03057, -3.87391,  3.6026],
                [-3.34315,  -2.53164,  -0.811431, -4.89399,  -3.8559,    6.88531,  -3.2929,  -3.72194, -3.27748,  2.88856],
                [-1.90005,  -1.09361,  -0.806441, -0.97338,  -2.32595,   -1.11973,  8.66179, -1.32893, -2.40079,  -1.61032],
                [-0.423727, -0.483017,  0.0594637,-1.58889,  -1.6448,    -3.66335, -1.96629, 7.33514,  -2.46024,  -3.59942],
                [-0.429409, -0.64549,   0.216125, -0.467977, -2.22157,   -3.08973, -3.47571, -2.39852, 6.63769,   -3.3078],
                [-1.03621,  2.04463,   -3.08081,  3.98603,   -0.0474521, -2.02132, -2.03285, -1.25411, -0.149985, -0.637863],
                [-1.20794,  -1.77778,   0.569958, 1.25975,   4.07279,    1.10171,  0.544073, 0.237209, 0.0408736, 2.0717]]
    else
        return [[-22.7402, -26.4621, 3.72192, 25.3496, 5.13677, -3.62378, 0.135073, -5.07247, -15.3604, 20.2022],
                [-18.4572, -10.6218, -7.83738, -29.8355, -6.09074, 5.45148, -1.21863, -1.7972, -0.930802, 31.2907],
                [-6.32903, -3.10829, -3.22149, -21.2094, -23.9865, -7.37753, 3.41425, -3.26152, -19.7259, 13.8982],
                [0.193882, 0.711791, -0.515013, -3.44767, -13.7064, -5.76557, -3.41942, 7.07691, -2.31784, -2.66733],
                [2.99406, 0.971162, 2.0226, 3.16372, 0.614502, -5.94997, -9.99085, -1.91549, 5.82236, -11.2777],
                [-0.0641174, 2.46199, -2.52839, 4.67603, 1.72256, -1.0436, -9.75621, -0.309607, 0.445067, -3.26586],
                [-1.94081, -2.57431, 0.631089, 0.918652, 2.32633, 0.241241, -0.1536, -0.317775, -0.414519, 3.99287]]
    end
end


function make_predictions!(ϕ::Array{Float64, 1}, weights::Array{Array{Float64, 1}, 1}, preds::Array{Float64, 1}; activate=Main.sigmoid)
    # preds .= sigmoid.(dot.([ϕ], weights))
    @inbounds for gvf = 1:length(weights)
        preds[gvf] = activate(dot(ϕ, weights[gvf]))
    end
end

end

module Chain

using LinearAlgebra
using GVFN.Environments

const weight_dim = (6, 9)
const ORACLE = [[0.0, 0.0, 0.0, 0.0, 0.0, 1.0], # > GVF 1 prediction when GVF 2
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]]

function get_optimal_weights(bptt=false)
    # return [zeros(9) for i in 1:6]
    return [[-4.33676  -7.14525   2.80853   5.51171   6.93764  -4.34167  -4.60836  -4.38966  -4.53675],
            [-2.23758  -1.65031 -0.587287  -2.98073  -2.64381   7.81357  -2.53582  -3.02891  -2.66565],
            [-2.68823  -1.46023  -1.22803  -1.12783  -2.96409  -1.28577   8.67141  -1.18291  -1.66096],
            [-2.02186  -1.20132 -0.820474  -2.12946  -2.42345  -5.01716  -2.12544   7.47598  -2.05409],
            [-2.08848  -1.79541 -0.293028  -0.90143  -2.72756  -2.91262  -4.13023  -2.42207   6.88209],
            [-1.17957   1.67837  -2.85797    3.8757 -0.617985  -1.83165  -1.95144  -1.76069 -0.609737]]
end

function get_parameters!(r, γ_t, ρ_t, obs, preds_tilde)
    r[1] = obs[2]
    for i in 2:(length(preds_tilde)-1)
        r[i] = preds_tilde[i-1]
    end
    # r[numgvfs] = cycleobs[agent_state+1][2]
    # γ_t[numgvfs] = 0.9*(1-r[1])
end

@inline sqr_error(env::CycleWorld.Environment, preds::Array{Float64}) = (preds .- ORACLE[env.pos+1]).^2


function make_predictions!(ϕ::Array{Float64, 1}, weights::Array{Array{Float64, 1}, 1}, preds::Array{Float64, 1}; activate=Main.sigmoid)
    # preds .= sigmoid.(dot.([ϕ], weights))
    @inbounds for gvf = 1:length(weights)
        preds[gvf] = activate(dot(ϕ, weights[gvf]))
    end
end

end

end

