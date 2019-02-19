

# using GVFN
using ProgressMeter
using LinearAlgebra
using Statistics
using GVFN.Environments
using GVFN
using StatsBase
using Random

# const e = exp(1)
# sigmoid(x::Float64) = (x / (1 + abs(x)))
sigmoid(x::Float64) = 1.0/(1.0 + ℯ^(-x))
function sigmoidprime(x::Float64)
    # sigmoid(x)*(1.0-sigmoid(x))
    tmp = sigmoid(x)
    return tmp*(1.0 - tmp)
end

clip(x::Float64) = clamp(x, 0.0, 1.0)
clipprime(x::Float64) = 1.0


struct LayerTransfer
    func!
    num_features
end


function make_features!(ϕ, obs, action, preds)
    fill!(ϕ, 0.0)

    place = (action-1)*(8*5 + 1 + 6) + 1
    ϕ[place] = 1

    place += 1
    ϕ[place:(place+5)] = obs

    place += 6
    ϕ[place:(place+(length(preds) - 1))] = preds

    return ϕ
end

function make_features_eval!(ϕ, obs, action, preds)
    fill!(ϕ, 0.0)
    # place = (action - 1)*(5)
    place = 1
    ϕ[place] = 1

    place += 1
    ϕ[place:(place+(length(preds) - 1))] = obs

    # ϕ[place:(place+(length(preds) - 1))] = preds
    return ϕ
end

struct GVFNetwork_Layers
    gvf_layers::Array{GVFNetwork, 1}
    layer_transfers::Array{LayerTransfer, 1}
    model_layers::Array{Networks.Network, 1}
    
end

function GVFNetwork_Layers(gvf_layers::Array{GVFNetwork, 1}, layer_transfers::Array{LayerTransfer, 1})
    model_layers = [Networks.Network(length(gvf_layers[layer]), layer_transfers[layer].num_features) for layer in 1:length(gvf_layers)]
    new(gvf_layers, layer_transfers, model_layers)
end

function run_gvfn(numsteps; α=0.1, α_2=0.1, λ=0.9, pertubation=nothing, ORACLE=nothing, gvfn=Networks.CycleWorld.GammaChain, bptt=false)

    rng = Random.GLOBAL_RNG

    env = CompassWorld


    env_state = env.init(rng)

    # agent_state = 0(8*5, (8*5 + 1 + 6)*3)
    numgvfs = Networks.CompassWorld.Rafols.num_gvfs
    numweights = (8*5 + 1 + 6)*3

    numgvfs_2 = 5
    numweights_2 = (8*5 + 1)

    gvfn = GVFNetwork_Layers(
        [
            Networks.CompassWorld.Rafols.build_gvfn(),
            Networks.CompassWorld.EvaluationLayer.build_gvfn()
        ],
        [
            LayerTransfer(make_features!, (8*5 + 1 + 6)*3),
            LayerTransfer(make_features_eval!, (8*5 + 1))
        ],
        [
            Networks.Network(numgvfs, numweights),
            Networks.Network(num_gvfs_2, numweights_2)
        ]
    )


    # initialize model
    # weights = [randn(rng, numweights) for gvf in 1:numgvfs]
    # weights = [zeros(numweights) for gvf in 1:numgvfs]
    # traces = [zeros(numweights) for gvf in 1:numgvfs]

    # network = Networks.Network([randn(rng, numweights) for gvf in 1:numgvfs], [zeros(numweights) for gvf in 1:numgvfs])
    # network = Networks.Network([zeros(numweights) for gvf in 1:numgvfs], [zeros(numweights) for gvf in 1:numgvfs])
    # network_2 = Networks.Network([zeros(numweights_2) for gvf in 1:numgvfs_2], [zeros(numweights_2) for gvf in 1:numgvfs_2])


    behaviour_policy_rand = [0.5, 0.25, 0.25]
    behaviour_policy_leap = [1.0, 0.0, 0.0]
    behaviour_policy_weightvec = Weights(behaviour_policy_rand)
    # network = Networks.Network(weights, traces)

    # initialize feature vectors
    ϕ_t = zeros(numweights)
    ϕ_tp1 = zeros(numweights)

    ϕ_t_2 = zeros(numweights_2)
    ϕ_tp1_2 = zeros(numweights_2)

    # initialize parameters
    r = zeros(numgvfs)
    γ_t = zeros(numgvfs)
    γ_tp1 = zeros(numgvfs)
    ρ_t = ones(numgvfs)
    ρ_tp1 = ones(numgvfs)

    preds = zeros(numgvfs)
    preds_tilde = zeros(numgvfs)

    r_2 = zeros(numgvfs_2)
    γ_t_2 = zeros(numgvfs_2)
    γ_tp1_2 = zeros(numgvfs_2)
    ρ_t_2 = ones(numgvfs_2)
    ρ_tp1_2 = ones(numgvfs_2)

    preds_2 = zeros(numgvfs_2)
    preds_tilde_2 = zeros(numgvfs_2)

    obs = env.get_observation(env_state)
    action = 1

    # make_features!(ϕ_t, obs, action, preds)
    gvfn.layer_transfers[1].func!(ϕ_tp1, obs, action, preds)
    gvfn.layer_transfers[2].func!(ϕ_tp1, preds, action, preds_2)
    # make_features_2!(ϕ_t_2, obs, preds, action, preds_2)

    gvfn.gvf_layers[1].mod.get_parameters!(gvfn_obj, r, γ_t, ρ_t, obs, preds_tilde, behaviour_policy_leap, action)
    gvfn.gvf_layers[2].mod.get_parameters!(gvfn_obj_2, r_2, γ_t_2, ρ_t_2, [preds; obs], preds_tilde_2, behaviour_policy_leap, action)

    pred_strg = zeros(numsteps, numgvfs)
    err_strg = zeros(numsteps, 5)

    pred_strg_2 = zeros(numsteps, numgvfs_2)
    # oracle_strg_2 = zeros(numsteps, numgvfs_2)
    err_strg_2 = zeros(numsteps, 5)

    Networks.make_predictions!(ϕ_t, gvfn.model_layers[1], preds; activate=clip)
    Networks.make_predictions!(ϕ_t_2, gvfn.model_layers[2], preds_2; activate=clip)


    option_weights = Weights([0.8, 0.2])
    # ϕ_tp1_view_obs = view(ϕ_tp1, 1:3)
    # ϕ_tp1_view_preds = view(ϕ_tp1, 4:length(ϕ_tp1))
    leap = false
    behaviour_policy = [0.0, 0.0, 0.0]
    @showprogress 0.1 "Steps: " for step in 1:numsteps
        if leap
            if obs[end] == 0.0
                leap = false
            end
        end

        if leap
            action = 1
        else
            opt = sample(rng, 1:2, option_weights)
            if opt == 1
                leap = false
                action = sample(rng, 1:3, behaviour_policy_weightvec)
                behaviour_policy = behaviour_policy_rand
            elseif opt == 2
                leap = true
                action = 1
                behaviour_policy = behaviour_policy_leap
            end
        end


        env_state, _, _ = env.step!(env_state, action)
        obs .= env.get_observation(env_state)

        # make_features!(ϕ_tp1, obs, action, preds)
        gvfn.layer_transfers[1].func!(ϕ_tp1, obs, action, preds)

        Networks.make_predictions!(ϕ_tp1, gvfn.model_layers[1], preds_tilde; activate=clip)
        gvfn.gvf_layers[1].get_parameters!(gvfn_obj, r, γ_tp1, ρ_t, obs, preds_tilde, behaviour_policy, action)

        Networks.optimize_gvfs!(gvfn.model_layers[1], preds, preds_tilde, ϕ_t, ϕ_tp1, r, γ_t, γ_tp1, ρ_t, α, λ, numgvfs, clipprime)
        Networks.make_predictions!(ϕ_tp1, gvfn.model_layers[1], preds; activate=clip)

        gvfn.gvf_layers[1].mod.sqr_error!(env_state, preds, view(err_strg, step, :))
        pred_strg[step, :] .= preds

        gvfn.layer_transfers[2].func!(ϕ_tp1_2, preds, action, preds_2)

        Networks.make_predictions!(ϕ_tp1_2, gvfn.model_layers[2], preds_tilde_2; activate=clip)
        gvfn.gvf_layers[2].get_parameters!(gvfn_obj_2, r_2, γ_tp1_2, ρ_t_2, [preds; obs], preds_tilde_2, behaviour_policy, action)

        Networks.optimize_gvfs!(gvfn.model_layers[2], preds_2, preds_tilde_2, ϕ_t_2, ϕ_tp1_2, r_2, γ_t_2, γ_tp1_2, ρ_t_2, α_2, λ, numgvfs_2, sigmoidprime)
        Networks.make_predictions!(ϕ_tp1_2, gvfn.model_layers[2], preds_2; activate=clip)

        gvfn.gvf_layers[1].mod.sqr_error!(env_state, preds_2, view(err_strg_2, step, :))
        pred_strg_2[step, :] .= preds_2
        # oracle_strg_2[step,:] .= .ORACLE(env_state)


        # make_features_2!(ϕ_tp1_2, obs, preds, action, preds_2)
        # Networks.make_predictions!(ϕ_tp1_2, network_2, preds_tilde_2; activate=sigmoid)
        # gvfn_2.get_parameters!(gvfn_obj_2, r_2, γ_tp1_2, ρ_t_2, [preds; obs], preds_tilde_2, behaviour_policy, action)

        # Networks.optimize_gvfs!(network_2, preds_2, preds_tilde_2, ϕ_t_2, ϕ_tp1_2, r_2, γ_t_2, γ_tp1_2, ρ_t_2, α_2, λ, numgvfs_2, sigmoidprime)
        # Networks.make_predictions!(ϕ_tp1_2, network_2, preds_2; activate=sigmoid)

        # gvfn_2.sqr_error!(env_state, preds_2, view(err_strg_2, step, :))
        # pred_strg_2[step, :] .= preds_2
        # oracle_strg_2[step,:] .= gvfn_2.ORACLE(env_state)

        ϕ_t .= ϕ_tp1
        γ_t .= γ_tp1

        ϕ_t_2 .= ϕ_tp1_2
        γ_t_2 .= γ_tp1_2

    end

    return err_strg, pred_strg, err_strg_2, pred_strg_2
end
