module GVFN

import Base.length


include("Environments.jl")

export Policy,
    Cumulant,
    Continuation,
    GVF,
    CumulantTypes,
    ContinuationTypes,
    PolicyTypes

include("GVF.jl")

end
