using GVFN: CycleWorld, step!, start!
using GVFN
using Flux
using Flux.Tracker
using Statistics
import LinearAlgebra.Diagonal
using Random
# using ProgressMeter
using FileIO
using ArgParse
using Random


function arg_parse(as::ArgParseSettings = ArgParseSettings())

    #Experiment
    @add_arg_table as begin
        "--seed"
        help="Seed of rng"
        arg_type=Int64
        default=0
        "--steps"
        help="number of steps"
        arg_type=Int64
        default=100
        "--savefile"
        help="save file for experiment"
        arg_type=String
        default="temp.jld"
        "--verbose"
        action=:store_true
    end


    #Cycle world
    @add_arg_table as begin
        "--size"
        help="The size of the compass world chain"
        arg_type=Int64
        default=8
    end

    # Algorithms
    @add_arg_table as begin
        "--alg"
        help="Algorithm"
        default="TDLambda"
        "--luparams"
        help="Parameters"
        arg_type=Float64
        default=[0.5, 0.9]
        nargs='+'
        "--truncation", "-t"
        help="Truncation parameter for bptt"
        arg_type=Int64
        default=1
        "--opt"
        help="Optimizer"
        default="Descent"
        "--optparams"
        help="Parameters"
        arg_type=Float64
        default=[0.5, 0.9]
        nargs='+'
        "--horde"
        help="The horde used for training"
        default="gamma_chain"
        "--gamma"
        help="The gamma value for the gamma_chain horde"
        arg_type=Float64
        default=0.9
    end

    return as
end


function rafols()
    
    cwc = GVFN.CompassWorldConst
    gvfs = Array{GVF, 1}()
    for color in 1:5
        new_gvfs = [GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.FORWARD)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(FeatureCumulant(color), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 4), ConstantDiscount(0.0), PersistentPolicy(cwc.LEFT)),
                    GVF(PredictionCumulant(8*(color-1) + 4), ConstantDiscount(0.0), PersistentPolicy(cwc.RIGHT)),
                    GVF(PredictionCumulant(8*(color-1) + 5), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)),
                    GVF(PredictionCumulant(8*(color-1) + 6), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD))]
        append!(gvfs, new_gvfs)
    end
    return Horde(gvfs)
end

function forward()
    cwc = GVFN.CompassWorldConst
    gvfs = [GVF(FeatureCumulant(color), StateTerminationDiscount(1.0, ((env_state)->env_state[cwc.WHITE] == 0)), PersistentPolicy(cwc.FORWARD)) for color in 1:5]
    return Horde(gvfs)
end

function oracle(env::CompassWorld, horde_str)
    cwc = GVFN.CompassWorldConst
    state = env.agent_state
    ret = Array{Float64,1}()
    if horde_str == "forward"
        ret = zeros(5)
        if state.dir == cwc.NORTH
            ret[cwc.ORANGE] = 1
        elseif state.dir == cwc.SOUTH
            ret[cwc.RED] = 1
        elseif state.dir == cwc.WEST
            if state.y == 1
                ret[cwc.GREEN] = 1
            else
                ret[cwc.BLUE] = 1
            end
        elseif state.dir == cwc.EAST
            ret[cwc.YELLOW] = 1
        else
            println(state.dir)
            throw("Bug Found in Oracle:Forward")
        end
    elseif horde_str == "rafols"
        throw("Not Implemented...")
    else
        throw("Bug Found in Oracle")
    end

    return ret
end

function get_action(state, env_state, rng=Random.GLOBAL_RNG)

    if state == ""
        state = "Random"
    end

    cwc = GVFN.CompassWorldConst
    

    if state == "Random"
        r = rand(rng)
        if r > 0.9
            state = "Leap"
        end
    end

    if state == "Leap"
        if env_state[cwc.WHITE] == 0.0
            state = "Random"
        else
            return state, (cwc.FORWARD, 1.0)
        end
    end
    r = rand(rng)
    if r < 0.2
        return state, (cwc.RIGHT, 0.2)
    elseif r < 0.4
        return state, (cwc.LEFT, 0.2)
    else
        return state, (cwc.FORWARD, 0.6)
    end
end

function get_action(rng=Random.GLOBAL_RNG)
    
    cwc = GVFN.CompassWorldConst

    r = rand(rng)
    if r < 0.2
        return cwc.RIGHT, 0.2
    elseif r < 0.4
        return cwc.LEFT, 0.2
    else
        return cwc.FORWARD, 0.6
    end
end

build_features(state) = [[1.0]; state; 1.0.-state]


mutable struct GVFALayer{F, A, V, T<:GVFN.AbstractGVF} <: GVFN.AbstractGVFLayer
    σ::F
    Wx::A
    Wh::A
    h::V
    horde::Horde{T}
end

GVFALayer(num_gvfs, num_actions, num_ext_features, horde; init=Flux.glorot_uniform, σ_int=Flux.σ) =
    GVFALayer(
        σ_int,
        # param(init(num_gvfs, num_actions*num_ext_features)),
        # param(init(num_gvfs, num_actions*num_gvfs)),
        init(num_gvfs, num_actions*num_ext_features),
        init(num_gvfs, num_actions*num_gvfs),
        # cumulants,
        # discounts,
        Flux.zeros(num_gvfs),
        horde)

Flux.hidden(m::GVFALayer) = m.h
Flux.@treelike GVFALayer

mutable struct TDLambda_test
    α::Float64
    λ::Float64
    γ_t::IdDict
    e::IdDict
    TDLambda_test(α, λ) = new(α, λ, IdDict(), IdDict())
end

function train!(gvfn::GVFALayer, lu::TDLambda_test, action_tm1, state_t, action_t, state_tp1, env_state_t, env_state_tp1, hidden_state_init, b_prob)
    
    λ = lu.λ
    α = lu.α

    ϕx_t = zeros(size(gvfn.Wx)[2])
    ϕh_t = zeros(size(gvfn.Wh)[2])
    num_feat_x_act = Int64(length(ϕx_t)/3)
    num_feat_h_act = Int64(length(ϕh_t)/3)
    # prinln(length(ϕx_t[((action_tm1-1)*num_feat_x_act + 1):((action_tm1)*num_feat_x_act)]))
    ϕx_t[((action_tm1-1)*num_feat_x_act + 1):((action_tm1)*num_feat_x_act)] .= state_t
    ϕh_t[((action_tm1-1)*num_feat_h_act + 1):((action_tm1)*num_feat_h_act)] .= hidden_state_init

    preds_t = Flux.data(gvfn.σ.(gvfn.Wx*ϕx_t + gvfn.Wh*ϕh_t))
    if gvfn.σ == Flux.σ
        preds_prime_t = preds_t.*(1.0 .- preds_t)
    else
        preds_prime_t = ones(length(preds_t))
    end
    # preds_prime_t = preds_t.*(1.0 .- preds_t)

    ϕx_tp1 = zeros(size(gvfn.Wx)[2])
    ϕh_tp1 = zeros(size(gvfn.Wh)[2])
    ϕx_tp1[((action_t-1)*num_feat_x_act + 1):((action_t)*num_feat_x_act)] .= state_tp1
    ϕh_tp1[((action_t-1)*num_feat_h_act + 1):((action_t)*num_feat_h_act)] .= Flux.data(preds_t)

    preds_tp1 = Flux.data(gvfn.σ.(gvfn.Wx*ϕx_tp1 + gvfn.Wh*ϕh_tp1))
    
    # Get t+1
    c, γ_tp1, π_prob = get(gvfn.horde, env_state_t, action_t, env_state_tp1, preds_tp1)
    γ = get!(lu.γ_t, gvfn, zeros(Float32, size(γ_tp1)...))::Array{Float32, 1}

    # _, γ_tp1, _ = get(gvfn.horde, env_state_tp1, action_t, env_state_tp1, preds_tp1)
    # ρ = clamp.(π_prob./b_prob, 0.0, 1.0)
    ρ = π_prob./b_prob
    # println(ρ)
    # println(ρ[((8*collect(0:4)).+4)])

    δ = (c + γ_tp1.*preds_tp1) - preds_t

    prms = params(gvfn)
    # println("train!")
    # grads = gradient(()->sum(δ), prms)
    # println(length(prms))
    # println(size.(prms))

    for gvf in 1:length(gvfn.horde)
        # println(size((α*ρ[gvf]*δ[gvf]*preds_prime_t[gvf]).*ϕx_t))
        # println(size(gvfn.Wx[gvf, :]))
        gvfn.Wx[gvf, :] .+= (α*ρ[gvf]*δ[gvf]*preds_prime_t[gvf]).*ϕx_t
        gvfn.Wh[gvf, :] .+= (α*ρ[gvf]*δ[gvf]*preds_prime_t[gvf]).*ϕh_t
    end


    # for weights in prms

    #     e = get!(lu.e, weights, zero(weights)::typeof(Flux.data(weights)))
    #     # println(size(weights), " ", size(e))
    #     # println(weights == gvfn.Wx)
    #     if size(weights)[2] == 39
    #         # println("Here")
    #         e .= ρ.*(convert(Array{Float64, 2}, Diagonal(γ)) * λ * e .+ preds_prime_t*ϕx_t')
    #     else
    #         # println(size(weights), " ", size(e), " ", size(preds_prime_t*ϕh_t'))
    #         e .= ρ.*(convert(Array{Float64, 2}, Diagonal(γ)) * λ * e .+ preds_prime_t*ϕh_t')
    #     end
    #     # println(grads[weights])
    #     # e .= ρ.*(convert(Array{Float64, 2}, Diagonal(γ)) * λ * e - Flux.data(grads[weights]))
    #     # e .= ρ.*(convert(Array{Float64, 2}, Diagonal(γ)) * λ * e + δ*ϕ)
    #     Flux.Tracker.update!(weights, α*e.*(δ))
    #     # weights .+= α*e.*δ
    # end

    γ .= γ_tp1

end




function main_experiment(args::Vector{String})

    as = arg_parse()
    parsed = parse_args(args, as)

    savefile = parsed["savefile"]
    savepath = dirname(savefile)
    # println(args)
    # println(savefile)

    if savepath != ""
        if !isdir(savepath)
            mkpath(savepath)
        end
    end

    num_steps = parsed["steps"]
    seed = parsed["seed"]
    rng = Random.MersenneTwister(seed)

    env = CompassWorld(parsed["size"], parsed["size"])
    num_state_features = get_num_features(env)
    horde = rafols()

    num_gvfs = length(horde)

    # alg_string = parsed["alg"]
    # gvfn_lu_func = getproperty(GVFN, Symbol(alg_string))
    # lu = gvfn_lu_func(Float64.(parsed["luparams"])...)
    # τ=parsed["truncation"]
    println("Here")
    lu = TDLambda_test(Float64(parsed["optparams"][1]), Float64(parsed["luparams"][1]))
    τ = 1

    # opt_string = parsed["opt"]
    # opt_func = getproperty(Flux, Symbol(opt_string))
    # opt = opt_func(Float64.(parsed["optparams"])...)

    pred_strg = zeros(num_steps, num_gvfs)
    err_strg = zeros(num_steps, 5)
    out_pred_strg = zeros(num_steps, 5)
    out_err_strg = zeros(num_steps, 5)

    out_horde = forward()
    out_opt = Descent(0.1)
    out_lu = TD()

    gvfn = GVFALayer(num_gvfs, 3, (6*2+1), horde; init=(dims...)->0.0001.*(rand(rng, Float64, dims...).-0.5), σ_int=Flux.σ)
    # gvfn = GVFALayer(num_gvfs, 3, (6*2+1), horde; init=(dims...)->0.01*rand(rng, Float32, dims...), σ_int=Flux.σ)
    model = Chain(StopGradient(gvfn), Flux.Dense(num_gvfs, length(out_horde), Flux.σ))

    _, s_t = start!(env)
    state_list = [(1, zeros(6)) for t in 1:τ]
    popfirst!(state_list)
    action_state = ""
    action_state, a_tm1 = get_action(action_state, s_t, rng)
    push!(state_list, (a_tm1[1], build_features(s_t)))
    hidden_state_init = zeros(num_gvfs)

    

    for step in 1:num_steps
        if parsed["verbose"]
            print(step, "\n")
        else
            if step % 10000 == 0
                print(step, "\r")
            end
        end
        action_state, a_t = get_action(action_state, s_t, rng)
        # println(action_state)
        # a_t = get_action(rng)
        # println(a_t)

        _, s_tp1, _, _ = step!(env, a_t[1])

        if length(state_list) == (τ+1)
            popfirst!(state_list)
        end
        push!(state_list, (a_t[1], build_features(s_tp1)))

        train!(gvfn, lu, a_tm1[1], state_list[1][2], a_t[1], state_list[2][2], s_t, s_tp1, hidden_state_init, a_t[2])

        action_tm1 = a_tm1[1]
        action_t = a_t[1]

        ϕx_t = zeros(size(gvfn.Wx)[2])
        ϕh_t = zeros(size(gvfn.Wh)[2])
        num_feat_x_act = Int64(length(ϕx_t)/3)
        num_feat_h_act = Int64(length(ϕh_t)/3)
        ϕx_t[((action_tm1-1)*num_feat_x_act + 1):((action_tm1)*num_feat_x_act)] .= state_list[1][2]
        ϕh_t[((action_tm1-1)*num_feat_h_act + 1):((action_tm1)*num_feat_h_act)] .= hidden_state_init

        preds_t = gvfn.σ.(gvfn.Wx*ϕx_t + gvfn.Wh*ϕh_t)

        ϕx_tp1 = zeros(size(gvfn.Wx)[2])
        ϕh_tp1 = zeros(size(gvfn.Wh)[2])
        ϕx_tp1[((action_t-1)*num_feat_x_act + 1):((action_t)*num_feat_x_act)] .= state_list[2][2]
        ϕh_tp1[((action_t-1)*num_feat_h_act + 1):((action_t)*num_feat_h_act)] .= Flux.data(preds_t)

        preds_tp1 = Flux.data(gvfn.σ.(gvfn.Wx*ϕx_tp1 + gvfn.Wh*ϕh_tp1))

        # train!(gvfn, opt, lu, hidden_state_init, state_list, s_tp1, a_t[1], a_t[2])
        

        # train!(model, out_horde, out_opt, out_lu, state_list[end-1:end], s_tp1, τ==1 ? hidden_state_init : preds[end-2], a_t[1], a_t[2])


        # if τ == 1
        #     reset!(gvfn, hidden_state_init)
        # else
        #     reset!(gvfn, preds[end-2])
        # end
        # reset!(gvfn, preds[end-2])
        # out_preds = model.(state_list[end-1:end])

        pred_strg[step, :] .= preds_tp1
        err_strg[step, :] .= preds_tp1[((8 .* collect(0:4)).+4)] .- oracle(env, "forward")
        # out_pred_strg[step, :] .= out_preds[end].data
        # out_err_strg[step, :] .= out_pred_strg[step, :] .- oracle(env, "forward")
        if parsed["verbose"]
            println(((8 .* collect(0:4)).+4))
            println(preds_tp1[((8 .* collect(0:4)).+4)])
            println(oracle(env, "forward"))
            println(env)
        end
        
        s_t .= s_tp1
        hidden_state_init .= Flux.data(preds_t)
        a_tm1 = a_t
        
    end

    # results = Dict(["predictions"=>pred_strg, "error"=>err_strg])
    results = Dict(["predictions"=>pred_strg, "error"=>err_strg, "out_pred"=>out_pred_strg, "out_err_strg"=>out_err_strg])
    save(savefile, results)
    # return pred_strg, err_strg
end
