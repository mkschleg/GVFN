
module TimeSeriesUtils

using ..GVFN, Reproduce

function LinSpacing(lo,hi,N)
    Γ = collect(LinRange(lo,hi,N))
    return Horde(
        map(γ->GVF(NormalizedCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ),NullPolicy()), Γ)
    )
end

get_horde(lo,hi,N) = LinSpacing(lo,hi,N)
get_horde(parsed::Dict)=get_horde(parsed["gamma_low"],parsed["gamma_high"], parsed["num_gvfs"])

end
