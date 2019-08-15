
module CycleWorldUtils

using ..GVFN, Reproduce

# export settings!, onestep, chain, gamma_chain, get_horde, oracle

function env_settings!(as::ArgParseSettings)
    @add_arg_table as begin
        "--chain"
        help="The length of the cycle world chain"
        arg_type=Int64
        default=6
    end
end

function horde_settings!(as::ArgParseSettings, prefix::AbstractString="")
    add_arg_table(as,
                  "--$(prefix)gamma",
                  Dict(:help=>"The gamma value for the gamma_chain horde",
                       :arg_type=>Float64,
                       :default=>0.9),
                  "--$(prefix)horde",
                  Dict(:help=>"The horde used for training",
                       :default=>"gamma_chain"))
end

function onestep(chain_length::Integer)
    gvfs = [GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())]
    return Horde(gvfs)
end

function chain(chain_length::Integer, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), NullPolicy()) for i in 2:chain_length]]
    return Horde(gvfs)
end

function gamma_chain(chain_length::Integer, γ::AbstractFloat, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), NullPolicy()) for i in 2:chain_length];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), NullPolicy())]]
    return Horde(gvfs)
end

function gamma_chain_scaled(chain_length::Integer, γ::AbstractFloat, pred_offset::Integer=0)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1 + pred_offset), ConstantDiscount(0.0), NullPolicy()) for i in 2:chain_length];
            [GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ), NullPolicy())]]
    return Horde(gvfs)
end

function gammas(gms::Array{Float64, 1})
    gvfs = [GVF(FeatureCumulant(1), ConstantDiscount(γ), NullPolicy()) for γ in gms]
    return Horde(gvfs)
end

function gammas_scaled(gms::Array{Float64, 1})
    gvfs = [GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ), NullPolicy()) for γ in gms]
    return Horde(gvfs)
end

function gammas_term(gms::Array{Float64, 1})
    gvfs = [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), NullPolicy()) for γ in gms]
    return Horde(gvfs)
end

function gammas_aj()
    gms = 1.0 .- 2.0 .^ collect(-7:-1)
    return gammas(gms)
end

function gammas_aj_term()
    gms = 1.0 .- 2.0 .^ collect(-7:-1)
    return gammas_term(gms)
end

function gammas_aj_scaled()
    gms = 1.0 .- 2.0 .^ collect(-7:-1)
    return gammas_scaled(gms)
end

function single_gamma_scaled(γ::AbstractFloat)
    Horde([GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ), NullPolicy())])
end

function get_horde(horde_str::AbstractString, chain_length::Integer, gamma::AbstractFloat, pred_offset::Integer=0)
    horde = nothing
    if horde_str == "chain"
        horde = chain(chain_length, pred_offset)
    elseif horde_str == "gamma_chain"
        horde = gamma_chain(chain_length, gamma, pred_offset)
    elseif horde_str == "gamma_chain_scaled"
        horde = gamma_chain_scaled(chain_length, gamma, pred_offset)
    elseif horde_str == "onestep"
        horde = onestep(chain_length)
    elseif horde_str == "gammas"
        horde = gammas(collect(0.0:0.1:0.9))
    elseif horde_str == "gammas_term"
        horde = gammas_term(collect(0.0:0.1:0.9))
    elseif horde_str == "gammas_scaled"
        horde = gammas_scaled(collect(0.0:0.1:0.9))
    elseif horde_str == "gammas_aj"
        horde = gammas_aj()
    elseif horde_str == "gammas_aj_term"
        horde = gammas_aj_term()
    elseif horde_str == "gammas_aj_scaled"
        horde = gammas_aj_scaled()
    elseif horde_str == "single_gamma_scaled"
        horde = single_gamma_scaled(gamma)
    else
        throw("Horde not available $(horde_str)")
    end
    return horde
end

get_horde(parsed::Dict, prefix="", pred_offset::Integer=0) = get_horde(parsed["$(prefix)horde"], parsed["chain"], parsed["$(prefix)gamma"], pred_offset)

function oracle(env::CycleWorld, horde_str, γ=0.9)
    chain_length = env.chain_length
    state = env.agent_state
    ret = Array{Float64,1}()
    if horde_str == "chain"
        ret = zeros(chain_length)
        ret[chain_length - state] = 1
    elseif horde_str == "gamma_chain"
        ret = zeros(chain_length + 1)
        ret[chain_length - state] = 1
        ret[end] = γ^(chain_length - state - 1)
    elseif horde_str == "onestep"
        #TODO: Hack fix.
        tmp = zeros(chain_length + 1)
        tmp[chain_length - state] = 1
        ret = [tmp[1]]
    elseif horde_str == "gammas"
        ret = collect(0.0:0.1:0.9).^(chain_length - state - 1)
    else
        throw("Bug Found")
    end

    return ret
end

build_features_cycleworld(s) = Float32[1.0, s[1], 1-s[1]]

end
