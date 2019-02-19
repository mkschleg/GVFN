

# using GVFN
using ProgressMeter
using LinearAlgebra
using Statistics
using GVFN.Environments
using GVFN

# const e = exp(1)
# sigmoid(x::Float64) = (x / (1 + abs(x)))
sigmoid(x::Float64) = 1.0/(1.0 + ℯ^(-x))
function sigmoidprime(x::Float64)
    # sigmoid(x)*(1.0-sigmoid(x))
    tmp = sigmoid(x)
    return tmp*(1.0 - tmp)
end

function run_gvfn(numsteps; α=0.1, λ=0.9, pertubation=nothing, ORACLE=nothing, gvfn=Networks.CycleWorld.GammaChain, bptt=false)
    γ_const = 0.9

    env = CycleWorld
    gvfn = Networks.CycleWorld.GammaChain

    behaviour_policy = Policy([1.0])

    gvfn_obj = gvfn.build_gvfn()

    # initialize the environment
    env_state = env.init()

    # agent_state = 0
    numgvfs = gvfn.weight_dim[1]
    numweights = gvfn.weight_dim[2]

    # initialize model
    weights = [randn(numweights) for gvf in 1:numgvfs]
    traces = [zeros(numweights) for gvf in 1:numgvfs]

    network = Networks.Network([randn(numweights) for gvf in 1:numgvfs], [zeros(numweights) for gvf in 1:numgvfs])

    # network = Networks.Network(weights, traces)

    # initialize feature vectors
    ϕ_t = zeros(numweights)
    ϕ_tp1 = zeros(numweights)

    # initialize parameters
    r = zeros(numgvfs)
    γ_t = zeros(numgvfs)
    γ_tp1 = zeros(numgvfs)
    ρ_t = ones(numgvfs)
    ρ_tp1 = ones(numgvfs)

    preds = zeros(numgvfs)
    preds_tilde = zeros(numgvfs)

    obs = env.get_observation(env_state)
    ϕ_t[1:3] .= obs

    get_parameters!(gvfn_obj, r, γ_t, ρ_t, obs, 1, preds_tilde, behaviour_policy)

    pred_strg = zeros(numsteps, numgvfs)
    err_strg = zeros(numsteps, numgvfs)

    Networks.make_predictions!(ϕ_t, network, preds)

    ϕ_tp1_view_obs = view(ϕ_tp1, 1:3)
    ϕ_tp1_view_preds = view(ϕ_tp1, 4:length(ϕ_tp1))

    for step in 1:numsteps

        env_state, _, _ = env.step!(env_state, 0)
        obs .= env.get_observation(env_state)

        ϕ_tp1_view_obs .= obs
        ϕ_tp1_view_preds .= preds

        Networks.make_predictions!(ϕ_tp1, network, preds_tilde)
        get_parameters!(gvfn_obj, r, γ_tp1, ρ_tp1, obs, 1, preds_tilde, behaviour_policy)

        Networks.optimize_gvfs!(network, preds, preds_tilde, ϕ_t, ϕ_tp1, r, γ_t, γ_tp1, ρ_t, α, λ, numgvfs, sigmoidprime)
        Networks.make_predictions!(ϕ_tp1, network, preds)
        gvfn.sqr_error!(env_state, preds, view(err_strg, step, :))

        ϕ_t .= ϕ_tp1
        γ_t .= γ_tp1
        ρ_t .= ρ_tp1

    end

    return err_strg
end
