
module TimeSeriesUtils

using ..GVFN, Reproduce


function horde_settings!(as::ArgParseSettings, prefix::AbstractString="")
    add_arg_table(as,
                  "--gamma-selector",
                  Dict(:help=>"type of discount selector used",
                       :arg_type=>String,
                       :default=>"Powers2"),
                  "--max-exponent",
                  Dict(:help=>"largest N s.t. gamma=1.0-2^-N",
                       :arg_type=>Int,
                       :default=>7))
end




function Powers2(N::Int)
    Γ = map(i->1.0-2.0^{-i}, 1:N)
    gvfs = map(γ -> GVF(FeatureCumulant(1), ConstantDiscount(γ), NullPolicy()), Γ )
    return Horde(gvfs)
end

function get_horde(N::Int, prefix::String)
    if prefix=="gvfn"
        return Powers2(N)
    else
        return Horde(GVF(FeatureCumulant(1), ConstantDiscount(0),NullPolicy()))
    end
    throw(DomainError("get_horde called with invalid prefix: $(prefix)"))
end

get_horde(parsed::Dict) = get_horde(parsed["max-exponent"])

end
