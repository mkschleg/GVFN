
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
    kys = keys(parsed)
    if "normalizer" ∉ kys
        return Identity()
    end

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

function get_horde(parsed::Dict)
    horde_t = parsed["horde"]
    num_targets = parsed["num_targets"]
    if horde_t == "LinSpacing"
        lo, hi, N = parsed["gamma_low"], parsed["gamma_high"], parsed["num_gvfs"]

        Γ = Float32.(vcat([collect(LinRange(lo,hi,N)) for _=1:num_targets]...))
        return Horde(
            map(γ->GVF(NormalizedCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ),NullPolicy()), Γ)
        )
    elseif horde_t == "LinUnity"
        lo, hi, N = parsed["gamma_low"], parsed["gamma_high"], parsed["num_gvfs"]

        Γ = Float32.(vcat([collect(LinRange(lo,hi,N)) for _=1:num_targets]...))
        return Horde(
            map(γ->GVF(UnityCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ),NullPolicy()), Γ)
        )
    elseif horde_t == "Exponential"
        N = parsed["num_gvfs"]

        Γs=[1.0-2.0^-i for i=1:N]
        Γ = Float32.(vcat([Γs for _=1:num_targets]...))
        return Horde(
            map(γ->GVF(NormalizedCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ),NullPolicy()), Γ)
        )
    else
        throw(DomainError("Invalid horde type: $(horde_t)"))
    end
end

end # END TimeSeriesUtils
