

using ProgressMeter
import LinearAlgebra.Diagonal
using Statistics
using Flux
using Flux.Tracker

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


const num_gvfs = 7
const weight_dim = (7, 10)
const γ_const = 0.9

const ORACLE = [0 0 0 0 0 1 γ_const^5; # > GVF 1 prediction when GVF 2
                0 0 0 0 1 0 γ_const^4;
                0 0 0 1 0 0 γ_const^3;
                0 0 1 0 0 0 γ_const^2;
                0 1 0 0 0 0 γ_const^1;
                1 0 0 0 0 0 γ_const^0;]

function get_parameters(r, γ_t, ρ_t, cycleobs, agent_state, preds_tilde)
    r[1] = cycleobs[agent_state+1][2]
    for i in 2:(numgvfs-1)
        r[i] = preds_tilde[i-1]
    end
    r[numgvfs] = cycleobs[agent_state+1][2]
    γ_t[numgvfs] = 0.9*(1-r[1])
end

end



function run_gvfn(numsteps; α=0.0, λ=0.0, pertubation=nothing, ORACLE=nothing, gvfn=GammaChain, bptt=false)
    γ_const = 0.9

    env = CycleWorld
    gvfn = GammaChain

    # OPTIMAL_WEIGHTS_SIGMOID = gvfn.get_optimal_weights(bptt)

    ORACLE = gvfn.ORACLE

    agent_state = env.start()

    numgvfs = gvfn.num_gvfs
    num_features = gvfn.weight_dim[2]

    ϕ_t = zeros(num_features)
    ϕ_tp1 = zeros(num_features)
    r = zeros(numgvfs)
    γ_t = zeros(numgvfs)
    γ_tp1 = zeros(numgvfs)

    weights = param(zeros(gvfn.weight_dim))
    traces = param(zeros(gvfn.weight_dim))


    predict(ϕ) = σ.(weights*ϕ)
    predict_notrack(ϕ) = σ.(weights.data*ϕ)
    derive(ϕ) = ((1.0 .- predict_notrack(ϕ)).*predict_notrack(ϕ))
    loss(ϕ, y) = Flux.mse(predict(ϕ), y)
    optimizer = Descent(α)
    td_target(c, ϕ_t, ϕ_tp1, γ_tp1, gvfn) = c .+ γ_tp1.*predict_notrack(ϕ_tp1)

    preds = zeros(numgvfs)
    preds[end-1] = 0

    preds_tilde = zeros(numgvfs)

    println(length(ϕ_t), length(preds_tilde))
    ϕ_t[1:3] .= env.observation[agent_state+1]
    ϕ_t[4:end] .= preds_tilde
    # ϕ_t[4:end] .= gvfn.ORACLE[6, :]

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

    # make_predictions!(ϕ_t, weights, numgvfs, preds)
    preds .= predict_notrack(ϕ_t)

    # y = zeros(numgvfs)
    # δ_t = param(zeros(numgvfs))
    target = zeros(numgvfs)

    for step in 1:numsteps
        print("step: ", step, "\r")
        agent_state = (agent_state + 1) % 6

        ϕ_tp1[1:3] .= env.observation[agent_state+1]
        ϕ_tp1[4:end] .= preds #preds

        # make_predictions!(ϕ_tp1, weights, numgvfs, preds_tilde)
        preds_tilde .= predict_notrack(ϕ_tp1)

        r[1] = env.observation[agent_state+1][2]
        for i in 2:(numgvfs)
            r[i] = preds_tilde[i-1]
        end
        r[numgvfs] = env.observation[agent_state+1][2]
        γ_tp1[numgvfs] = 0.9*(1-r[1])

        # Flux.truncate!(weights)
        target .= td_target(r, ϕ_t, ϕ_tp1, γ_tp1, nothing)
        # Flux.train!(loss, [weights], [(ϕ_t, δ_t)], optimizer)
        
        y = predict(ϕ_t)
        δ_t = target - y
        grads = Tracker.gradient(() -> sum(δ_t), Params([weights]))
        traces = convert(Array{Float64, 2}, Diagonal(γ_t)) * λ * traces - grads[weights]
        Flux.Tracker.update!(weights, α.*traces.*(δ_t))

        preds .= predict_notrack(ϕ_tp1)
        pred_strg[step, :] .= preds
        err_strg[step, :] .= (preds - gvfn.ORACLE[agent_state+1,:]).^2
        loss_strg[step, :] .= (r .+ γ_tp1.*preds_tilde .- preds).^2
        loss_oracle_strg[step, :] .= (r .+ γ_tp1.*gvfn.ORACLE[((agent_state + 1)%6) + 1] .- preds).^2

        ϕ_t .= ϕ_tp1
        γ_t .= γ_tp1

        # if step % 50000 == 0
        #     GC.gc();
        # end
    end
    return pred_strg, err_strg, loss_strg, loss_oracle_strg
end

