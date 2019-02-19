

# using GVFN
using ProgressMeter
using LinearAlgebra
using UnsafeArrays

sigmoid(x::Float64) = 1.0/(1.0 + exp(-x))
sigmoidprime(x::Float64) = sigmoid(x)*(1.0-sigmoid(x))

function optimize_gvfs!(preds::Array{Float64, 1},
                        preds_tilde::Array{Float64, 1},
                        weights::Array{Float64, 2},
                        traces::Array{Float64, 2},
                        ϕ_t::Array{Float64, 1},
                        ϕ_tp1::Array{Float64, 1},
                        r::Array{Float64, 1},
                        γ_t::Array{Float64, 1},
                        γ_tp1::Array{Float64, 1},
                        λ::Float64,
                        α::Float64,
                        numgvfs::Integer,
                        δ::Array{Float64, 1},
                        preds_prime::Array{Float64, 1})
    # Threads.@threads for gvf in 1:numgvfs
    @uviews weights traces preds_prime preds ϕ_t ϕ_tp1 γ_tp1 γ_t δ r begin
        preds_prime .= sigmoidprime.(weights*ϕ_t)
        δ .= r .+ γ_tp1.*preds_tilde .- preds
        @inbounds for gvf in 1:numgvfs
            trace_view = view(traces, gvf, :)
            # δ[gvf] = r[gvf] + γ_tp1[gvf]*preds_tilde[gvf] - preds[gvf]
            trace_view .= 1.0.*(γ_t[gvf].*λ.*trace_view .+ preds_prime[gvf].*ϕ_t)
        end
        # weights .+= (α.*δ).*traces
        
        # δ .= r .+ γ_tp1.*preds_tilde .- preds
        # # traces .= traces.*(λ.*γ_t) .+ sigmoidprime.(weights*ϕ_t)*ϕ_t'
        # preds_prime = sigmoidprime.(weights*ϕ_t)
        # traces .= traces.*(λ.*γ_t)
        # # traces .+= sigmoidprime.(weights*ϕ_t)*ϕ_t'
        # @inbounds for gvf in 1:numgvfs
        #     traces[gvf,:] .+= preds_prime[gvf].*ϕ_t
        # end
        weights .+= traces.*(α.*δ)
    end
end


function make_predictions!(ϕ::Array{Float64, 1}, weights::Array{Float64, 2}, numgvfs::Int64, preds::Array{Float64, 1})
    # @uviews weights ϕ begin
    @views preds .= sigmoid.(weights*ϕ)
    # end
    # for gvf = 1:numgvfs
    #     preds[gvf] = sigmoid(dot(ϕ, weights[gvf]))
    # end
end

function test_mat(numsteps; pertubation=0.0)
    γ_const = 0.9

    # OPTIMAL_WEIGHTS::Array{Float64, 2} = [0 0 0 0 1 0 0 0 0 0; # > GVF 1 prediction when GVF 2
    #                                       0 0 0 0 0 1 0 0 0 0;
    #                                       0 0 0 0 0 0 1 0 0 0;
    #                                       0 0 0 0 0 0 0 1 0 0;
    #                                       0 0 0 0 0 0 0 0 1 0;
    #                                       0 1 0 0 0 0 0 0 0 0;
    #                                       0 0 0 0 0 0 0 0 0 0;]

    # OPTIMAL_WEIGHTS_SIGMOID = [[-22.7402, -26.4621, 3.72192, 25.3496, 5.13677, -3.62378, 0.135073, -5.07247, -15.3604, 20.2022],
    #                            [-18.4572, -10.6218, -7.83738, -29.8355, -6.09074, 5.45148, -1.21863, -1.7972, -0.930802, 31.2907],
    #                            [-6.32903, -3.10829, -3.22149, -21.2094, -23.9865, -7.37753, 3.41425, -3.26152, -19.7259, 13.8982],
    #                            [0.193882, 0.711791, -0.515013, -3.44767, -13.7064, -5.76557, -3.41942, 7.07691, -2.31784, -2.66733],
    #                            [2.99406, 0.971162, 2.0226, 3.16372, 0.614502, -5.94997, -9.99085, -1.91549, 5.82236, -11.2777],
    #                            [-0.0641174, 2.46199, -2.52839, 4.67603, 1.72256, -1.0436, -9.75621, -0.309607, 0.445067, -3.26586],
    #                            [-1.94081, -2.57431, 0.631089, 0.918652, 2.32633, 0.241241, -0.1536, -0.317775, -0.414519, 3.99287]]

    # ORACLE::Array{Float64, 2} = [0 0 0 0 0 1 γ_const^5; # > GVF 1 prediction when GVF 2
    #                              0 0 0 0 1 0 γ_const^4;
    #                              0 0 0 1 0 0 γ_const^3;
    #                              0 0 1 0 0 0 γ_const^2;
    #                              0 1 0 0 0 0 γ_const^1;
    #                              1 0 0 0 0 0 γ_const^0;]
    cycleobs = [[1.0, 1.0, 0.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 1.0]]
    agent_state = 0
    numgvfs = 100
    numweights = 3 + numgvfs

    ϕ_t = zeros(numweights)
    ϕ_tp1 = zeros(numweights)
    r = zeros(numgvfs)
    γ_t = zeros(numgvfs)
    γ_tp1 = zeros(numgvfs)
    δ = zeros(numgvfs)
    preds_prime = zeros(numgvfs)
    # weights = fill(0.0, (numgvfs, numweights))
    # weights = 0.001*rand(numgvfs, numweights)
    weights = 0.001*randn(numgvfs, numweights)
    traces = 0.001*zeros(numgvfs, numweights)
    # weights = [0.001*randn(numweights) for gvf in 1:numgvfs]
    # traces = [zeros(numweights) for gvf in 1:numgvfs]
    # for (weight_idx, weightvec) in enumerate(weights)
    #     weightvec .= OPTIMAL_WEIGHTS_SIGMOID[weight_idx]
    #     weightvec .+= pertubation*randn(length(weightvec))
    #     println(weightvec)
    # end

    # activate = sigmoid
    # activateprime = sigmoidprime

    preds = zeros(numgvfs)
    preds[end-1] = 0

    preds_tilde = zeros(numgvfs)

    ϕ_t[1:3] = cycleobs[agent_state+1]
    # ϕ_t[4:end] .= ORACLE[agent_state+1, :]

    r[1] = cycleobs[agent_state+1][2]
    for i in 2:(numgvfs-1)
        r[i] = preds_tilde[i-1]
    end
    r[numgvfs] = cycleobs[agent_state+1][2]
    # r[numgvfs] = 1/6

    γ_t[numgvfs] = 0.9*(1-r[1])

    pred_strg = zeros(numsteps, numgvfs)

    # preds .= sigmoid.(dot.([ϕ_t], weights))
    make_predictions!(ϕ_t, weights, numgvfs, preds)
    # println(preds)

    # @showprogress 1 "Step: " for step in 1:numsteps
    for step in 1:numsteps

        agent_state = (agent_state + 1) % 6

        ϕ_tp1[1:3] .= cycleobs[agent_state+1]
        ϕ_tp1[4:end] .= preds

        # preds_tilde .= sigmoid.(dot.([ϕ_tp1], weights))
        # for gvf = 1:numgvfs
        #     preds_tilde[gvf] .= sigmoid(dot(ϕ_tp1, weights[gvf]))
        # end
        make_predictions!(ϕ_tp1, weights, numgvfs, preds_tilde)
        # print(preds_tilde)
        r[1] = cycleobs[agent_state+1][2]
        for i in 2:(numgvfs)
            r[i] = preds_tilde[i-1]
        end
        r[numgvfs] = cycleobs[agent_state+1][2]
        γ_tp1[numgvfs] = 0.9*(1-r[1])

        α=0.2
        λ=0.9
        
        optimize_gvfs!(preds, preds_tilde, weights, traces, ϕ_t, ϕ_tp1, r, γ_t, γ_tp1, λ, α, numgvfs, δ, preds_prime)
        # Threads.@threads for gvf in 1:numgvfs
        #     δ = r[gvf] + γ_tp1[gvf]*preds_tilde[gvf] - preds[gvf]
        #     traces[gvf] .= 1.0.*((γ_t[gvf]*λ).*traces[gvf] .+ sigmoidprime(dot(weights[gvf], ϕ_t)).*ϕ_t)
        #     weights[gvf] .+= (α*δ).*traces[gvf]
        # end

        # preds .= sigmoid.(dot.([ϕ_tp1], weights))
        make_predictions!(ϕ_tp1, weights, numgvfs, preds)
        pred_strg[step, :] .= preds

        ϕ_t .= ϕ_tp1
        # # println(typeof(γ_t), " ", typeof(γ_tp1))
        γ_t .= γ_tp1

        # println( preds)
    end
    return pred_strg, weights
end



