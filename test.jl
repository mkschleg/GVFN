

using GVFN
using ProgressMeter
using LinearAlgebra


# OPTIMAL_WEIGHTS = [0 0 0 0 1 0 0 0 0 0 0;
#                    0 0 0 0 0 1 0 0 0 0 0;
#                    0 0 0 0 0 0 1 0 0 0 0;
#                    0 0 0 0 0 0 0 1 0 0 0;
#                    0 0 0 0 0 0 0 0 1 0 0;
#                    0 0 0 0 0 0 0 0 0 1 0;
#                    0 1 0 0 0 0 0 0 0 0 0;
#                    0 0 0 0 0 0 0 0 0 0 0;]


function test(numsteps; algorithm=LinearRL.TD!)

    cycleobs = [[1.0,1.0,0.0],
                [1.0,0.0,1.0],
                [1.0,0.0,1.0],
                [1.0,0.0,1.0],
                [1.0,0.0,1.0],
                [1.0,0.0,1.0],
                [1.0,0.0,1.0]]
    agent_state = 0
    numgvfs = 1000
    numweights = 3 + numgvfs

    ϕ_t = zeros(numweights)
    ϕ_tp1 = zeros(numweights)
    r = zeros(numgvfs)
    γ_t = zeros(numgvfs)
    γ_tp1 = zeros(numgvfs)
    # weights = fill(0.0, (numgvfs, numweights))
    # weights = 0.001*rand(numgvfs, numweights)
    weights = [zeros(numweights) for gvf in 1:numgvfs]
    # weights = convert(Array{Float64, 2}, OPTIMAL_WEIGHTS)
    traces = zeros(numgvfs, numweights)
    preds = zeros(numgvfs)
    preds[end-1] = 1

    preds_tilde = zeros(numgvfs)

    ϕ_t[1:3] = cycleobs[agent_state+1]

    r[1] = cycleobs[agent_state+1][2]
    for i in 2:(numgvfs-1)
        r[i] = preds[i-1]
    end
    r[numgvfs] = 0

    γ_t[numgvfs] = 0.9*(1-r[1])

    # pred_strg = zeros(numsteps, numgvfs)
    # state_strg = zeros(numsteps)

    @showprogress 1 "Step: " for step in 1:numsteps
    # for step in 1:numsteps
        # print(step, "\n")
        agent_state = (agent_state + 1) % 7
        # println(agent_state)
        ϕ_tp1[1:3] = cycleobs[agent_state+1]
        ϕ_tp1[4:end] = preds
        # println(size(weights[1]))
        # println(size(ϕ_tp1))
        # println(ϕ_tp1)
        preds_tilde .= clamp!(dot.([ϕ_tp1],weights), 0.0,1.0)
        # println(ϕ_tp1)
        # println(preds_tilde)

        r[1] = cycleobs[agent_state+1][2]
        for i in 2:(numgvfs-1)
            r[i] = preds_tilde[i-1]
        end
        r[numgvfs] = cycleobs[agent_state+1][2]
        γ_tp1[numgvfs] = 0.9*(1-r[1])

        # println(ϕ_t)
        # println(γ_t)
        # println(r)

        for gvf in 1:numgvfs
        # Threads.@threads for gvf in 1:numgvfs
            # function TD!(weights, α, ρ, state_t, state_tp1, reward, gamma, terminal)
            LinearRL.TD!(weights[gvf], 0.1, 1.0, ϕ_t, ϕ_tp1, r[gvf], γ_tp1[gvf], false)
            # LinearRL.TD!(view(weights, gvf, :), 0.1, 1.0, ϕ_t, ϕ_tp1, r[gvf], γ_t[gvf], false)
            # LinearRL.TDLambda!(view(weights, gvf, :), view(traces, gvf, :), 0.1, 1.0, ϕ_t, ϕ_tp1, r[gvf], γ_t[gvf], γ_tp1[gvf], 0.9, false)
        end

        preds .= clamp!(dot.([ϕ_tp1],weights), 0.0, 1.0)
        # pred_strg[step, :] = preds
        # state_strg[step] = agent_state
        ϕ_t .= ϕ_tp1
        γ_t .= γ_tp1

        # println(preds)
        

    end
    
end



