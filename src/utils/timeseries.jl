
module TimeSeriesUtils

using ..GVFN, Reproduce

function Powers2(N::Int)
    Γ = map(i->1.0-2.0^(-i), 1:N)
    gvfs = map(γ -> GVF(FeatureCumulant(1), ConstantDiscount(γ), NullPolicy()), Γ )
    return Horde(gvfs)
end

function get_horde(N::Int)
    return Powers2(N)
end

get_horde(parsed::Dict)=get_horde(parsed["max-exponent"])

end
