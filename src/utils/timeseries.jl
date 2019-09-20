
module TimeSeriesUtils

using ..GVFN, Reproduce

abstract type Normalizer end

struct  Identity <: Normalizer
end
(id::Identity)(x) = x

mutable struct IncrementalUnity{F} <: Normalizer
    mn::Vector{F}
    mx::Vector{F}
end

IncrementalUnity() = IncrementalUnity([0.0],[1.0])

function (u::IncrementalUnity)(x)
    u.mn, u.mx = min.(x,u.mn), max.(x,u.mx)
    return (x.-u.mn) ./ (u.mx.-u.mn)
end

struct Unity{F} <: Normalizer
    mn::F
    mx::F
end

(u::Unity)(x) = (x.-u.mn) ./ (u.mx.-u.mn)

function getNormalizer(parsed::Dict)
    n=parsed["normalizer"]
    if n=="Identity"
        return Identity()
    elseif n=="IncrementalUnity"
        return IncrementalUnity()
    elseif n=="Unity"
        return Unity(parsed["min"],parsed["max"])
    else
        throw(DomainError("Invalid normalizer given: $n"))
    end
end

function LinSpacing(lo,hi,N)
    Γ = collect(LinRange(lo,hi,N))
    return Horde(
        map(γ->GVF(NormalizedCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ),NullPolicy()), Γ)
    )
end

get_horde(lo,hi,N) = LinSpacing(lo,hi,N)
get_horde(parsed::Dict)=get_horde(parsed["gamma_low"],parsed["gamma_high"], parsed["num_gvfs"])

end
