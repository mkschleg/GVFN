# using GVFN
using ProgressMeter
using LinearAlgebra
using Statistics

sigmoid(x::Float64) = 1.0/(1.0 + exp(-x))
sigmoidprime(x::Float64) = sigmoid(x)*(1.0-sigmoid(x))

function optimize_gvfs!(preds::Array{Float64, 1},
                        preds_tilde::Array{Float64, 1},
                        weights::Array{Array{Float64, 1}, 1},
                        traces::Array{Array{Float64, 1}, 1},
                        ϕ_t::Array{Float64, 1},
                        ϕ_tp1::Array{Float64, 1},
                        r::Array{Float64, 1},
                        γ_t::Array{Float64, 1},
                        γ_tp1::Array{Float64, 1},
                        α::Float64,
                        λ::Float64,
                        numgvfs::Integer)
    # Threads.@threads for gvf in 1:numgvfs
    for gvf in 1:numgvfs
        δ = r[gvf] + γ_tp1[gvf]*preds_tilde[gvf] - preds[gvf]
        traces[gvf] .= 1.0.*((γ_t[gvf]*λ).*traces[gvf] .+ sigmoidprime(dot(weights[gvf], ϕ_t)).*ϕ_t)
        weights[gvf] .+= (α*δ).*traces[gvf]
    end
end

function make_predictions!(ϕ::Array{Float64, 1}, weights::Array{Array{Float64, 1}, 1}, numgvfs::Int64, preds::Array{Float64, 1})
    # preds .= sigmoid.(dot.([ϕ], weights))
    @inbounds for gvf = 1:numgvfs
        preds[gvf] = sigmoid(dot(ϕ, weights[gvf]))
    end
end


module CycleWorld
const observation = [[1.0, 1.0, 0.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 0.0, 1.0],
                     [1.0, 0.0, 1.0]]

function get_obs(agent_state)
    return observation[agent_state]
end

function step(agent_state, action)
    return agent_state + 1
end

function start()
    return 0
end

end


module GammaChain

const weight_dim = (7, 10)
const γ_const = 0.9

const ORACLE = [0 0 0 0 0 1 γ_const^5; # > GVF 1 prediction when GVF 2
                0 0 0 0 1 0 γ_const^4;
                0 0 0 1 0 0 γ_const^3;
                0 0 1 0 0 0 γ_const^2;
                0 1 0 0 0 0 γ_const^1;
                1 0 0 0 0 0 γ_const^0;]

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



function get_parameters(r, γ_t, ρ_t, cycleobs, agent_state, preds_tilde)
    r[1] = cycleobs[agent_state+1][2]
    for i in 2:(numgvfs-1)
        r[i] = preds_tilde[i-1]
    end
    r[numgvfs] = cycleobs[agent_state+1][2]
    γ_t[numgvfs] = 0.9*(1-r[1])
end

end

module Chain

const weight_dim = (7, 10)
const ORACLE = [0 0 0 0 0 1; # > GVF 1 prediction when GVF 2
                0 0 0 0 1 0;
                0 0 0 1 0 0;
                0 0 1 0 0 0;
                0 1 0 0 0 0;
                1 0 0 0 0 0;]

function get_optimal_weights(bptt=false)
    # return [zeros(9) for i in 1:6]
    return [[-4.33676  -7.14525   2.80853   5.51171   6.93764  -4.34167  -4.60836  -4.38966  -4.53675],
            [-2.23758  -1.65031 -0.587287  -2.98073  -2.64381   7.81357  -2.53582  -3.02891  -2.66565],
            [-2.68823  -1.46023  -1.22803  -1.12783  -2.96409  -1.28577   8.67141  -1.18291  -1.66096],
            [-2.02186  -1.20132 -0.820474  -2.12946  -2.42345  -5.01716  -2.12544   7.47598  -2.05409],
            [-2.08848  -1.79541 -0.293028  -0.90143  -2.72756  -2.91262  -4.13023  -2.42207   6.88209],
            [-1.17957   1.67837  -2.85797    3.8757 -0.617985  -1.83165  -1.95144  -1.76069 -0.609737]]
end

function get_parameters(r, γ_t, ρ_t, cycleobs, agent_state, preds_tilde)
    r[1] = cycleobs[agent_state+1][2]
    for i in 2:(numgvfs-1)
        r[i] = preds_tilde[i-1]
    end
    # r[numgvfs] = cycleobs[agent_state+1][2]
    # γ_t[numgvfs] = 0.9*(1-r[1])
end

end

function run_gvfn(numsteps; α=0.0, λ=0.0, pertubation=nothing, ORACLE=nothing, gvfn=GammaChain, bptt=false)
    γ_const = 0.9

    env = CycleWorld
    gvfn = GammaChain

    OPTIMAL_WEIGHTS_SIGMOID = gvfn.get_optimal_weights(bptt)

    ORACLE::Array{Float64, 2} = [0 0 0 0 0 1 γ_const^5; # > GVF 1 prediction when GVF 2
                                 0 0 0 0 1 0 γ_const^4;
                                 0 0 0 1 0 0 γ_const^3;
                                 0 0 1 0 0 0 γ_const^2;
                                 0 1 0 0 0 0 γ_const^1;
                                 1 0 0 0 0 0 γ_const^0;]

    agent_state = 0
    numgvfs = 7
    numweights = 3 + numgvfs

    ϕ_t = zeros(numweights)
    ϕ_tp1 = zeros(numweights)
    r = zeros(numgvfs)
    γ_t = zeros(numgvfs)
    γ_tp1 = zeros(numgvfs)

    weights = [zeros(numweights) for gvf in 1:numgvfs]
    traces = [zeros(numweights) for gvf in 1:numgvfs]
    if pertubation != nothing
        for (weight_idx, weightvec) in enumerate(weights)
            weightvec .= OPTIMAL_WEIGHTS_SIGMOID[weight_idx] .+ pertubation[weight_idx,:]
        end
    end

    preds = zeros(numgvfs)
    preds[end-1] = 0

    preds_tilde = zeros(numgvfs)

    ϕ_t[1:3] = env.observation[agent_state+1]
    ϕ_t[4:end] .= gvfn.ORACLE[6, :]

    r[1] = env.observation[agent_state+1][2]
    for i in 2:(numgvfs-1)
        r[i] = preds_tilde[i-1]
    end
    r[numgvfs] = env.observation[agent_state+1][2]
    γ_t[numgvfs] = 0.9*(1-r[1])

    pred_strg = zeros(numsteps, numgvfs)
    err_strg = zeros(numsteps, numgvfs)
    loss_strg = zeros(numsteps, numgvfs)
    loss_oracle_strg = zeros(numsteps, numgvfs)

    make_predictions!(ϕ_t, weights, numgvfs, preds)

    for step in 1:numsteps

         agent_state = (agent_state + 1) % 6

        ϕ_tp1[1:3] .= env.observation[agent_state+1]
        ϕ_tp1[4:end] .= preds

        make_predictions!(ϕ_tp1, weights, numgvfs, preds_tilde)

        r[1] = env.observation[agent_state+1][2]
        for i in 2:(numgvfs)
            r[i] = preds_tilde[i-1]
        end
        r[numgvfs] = env.observation[agent_state+1][2]
        γ_tp1[numgvfs] = 0.9*(1-r[1])

        optimize_gvfs!(preds, preds_tilde, weights, traces, ϕ_t, ϕ_tp1, r, γ_t, γ_tp1, α, λ, numgvfs)

        make_predictions!(ϕ_tp1, weights, numgvfs, preds)
        pred_strg[step, :] .= preds
        err_strg[step, :] .= (preds - gvfn.ORACLE[agent_state+1,:]).^2
        loss_strg[step, :] .= (r .+ γ_tp1.*preds_tilde .- preds).^2
        loss_oracle_strg[step, :] .= (r .+ γ_tp1.*gvfn.ORACLE[((agent_state + 1)%6) + 1] .- preds).^2

        ϕ_t .= ϕ_tp1
        γ_t .= γ_tp1

    end

    return err_strg, loss_strg, loss_oracle_strg
end



function get_loss_grid(dir1=nothing, dir2=nothing; resolution=0.1, distance=1, rand_dir=false, bptt=false)

    if rand_dir
        dir1 = randn(7,10)
        dir2 = randn(7,10)
    elseif dir1 == nothing && dir2 == nothing
        # dir1 = [randn(10) for i in 1:7]
        dir1 = randn(7,10)
        dir1 ./= sum(sum(dir1))
        dir2 = reshape(qr(reshape(dir1, (length(dir1)))).Q[2,:], (7,10))
    elseif dir1 == nothing
        dir1 = reshape(qr(reshape(dir2, (length(dir2)))).Q[2,:], (7,10))
    else
        dir2 = reshape(qr(reshape(dir1, (length(dir1)))).Q[2,:], (7,10))
    end


    alphas = collect(-distance:resolution:distance)

    results_err = zeros(length(alphas), length(alphas))
    results_loss = zeros(length(alphas), length(alphas))
    results_oracle_loss = zeros(length(alphas), length(alphas))

    @showprogress 0.1 "step: " for (α_idx, α) in enumerate(alphas)
        Threads.@threads for η_idx = 1:length(alphas)
        # for η_idx = 1:length(alphas)
            err_strg, loss_strg, loss_oracle_strg = run_gvfn(7; pertubation=(α.*dir1 .+ alphas[η_idx].*dir2), bptt=bptt)
            results_err[α_idx, η_idx] = mean(sqrt.(mean(err_strg; dims=2)))
            results_loss[α_idx, η_idx] = mean(sqrt.(mean(loss_strg; dims=2)))
            results_oracle_loss[α_idx, η_idx] = mean(sqrt.(mean(loss_oracle_strg; dims=2)))
        end
    end

    ret_dict = Dict("error"=>results_err, "loss"=>results_loss, "loss_oracle"=>results_oracle_loss)

    return ret_dict
end
