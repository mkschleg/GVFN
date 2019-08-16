
module TimeSeriesUtils

using ..GVFN, Reproduce

# function Powers2(N::Int)
#     Γ = map(i->1.0-2.0^(-i), 1:N)
#     gvfs = map(γ -> GVF(ScaledCumulant((1-γ),FeatureCumulant(1)), ConstantDiscount(γ), NullPolicy()), Γ )
#     return Horde(gvfs)
# end

function LinSpacing(lo,hi,N)
    Γ = collect(LinRange(lo,hi,N))
    return Horde(
        map(γ->GVF(ScaledCumulant(1-γ, FeatureCumulant(1)), ConstantDiscount(γ),NullPolicy()), Γ)
    )
end

get_horde(lo,hi,N) = LinSpacing(lo,hi,N)
get_horde(parsed::Dict)=get_horde(parsed["gamma_low"],parsed["gamma_high"], parsed["num_gvfs"])


end
