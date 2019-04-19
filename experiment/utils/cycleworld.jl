
module CycleWorldUtils

using GVFN, ArgParse

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

function chain(chain_length::Integer)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1), ConstantDiscount(0.0), NullPolicy()) for i in 2:chain_length]]
    return Horde(gvfs)
end

function gamma_chain(chain_length::Integer, γ::AbstractFloat)
    gvfs = [[GVF(FeatureCumulant(1), ConstantDiscount(0.0), NullPolicy())];
            [GVF(PredictionCumulant(i-1), ConstantDiscount(0.0), NullPolicy()) for i in 2:chain_length];
            [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), NullPolicy())]]
    return Horde(gvfs)
end

function gammas(gms::Array{Float64, 1})
    gvfs = [GVF(FeatureCumulant(1), StateTerminationDiscount(γ, ((env_state)->env_state[1] == 1)), NullPolicy()) for γ in gms]
    return Horde(gvfs)
end

function get_horde(horde_str::AbstractString, chain_length::Integer, gamma::AbstractFloat)
    horde = chain(chain_length)
    if horde_str == "gamma_chain"
        horde = gamma_chain(chain_length, gamma)
    elseif horde_str == "onestep"
        horde = onestep(chain_length)
    elseif horde_str == "gammas"
        horde = gammas(collect(0.0:0.1:0.9))
    end
    return horde
end

get_horde(parsed::Dict, prefix="") = get_horde(parsed["$(prefix)horde"], parsed["chain"], parsed["$(prefix)gamma"])

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

end
