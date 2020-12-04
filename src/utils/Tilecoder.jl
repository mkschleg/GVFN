

module TileCoding

mutable struct IHT
    size::Integer
    overfullCount::Int64
    dictionary::Dict{Array{Int64, 1}, Int64}
    IHT(sizeval) = new(sizeval, 0, Dict{Array{Int64, 1}, Int64}())
end

capacity(iht::IHT) = iht.size
count(iht::IHT) = length(iht.dictionary)
fullp(iht::IHT) = length(iht.dictionary) >= count(iht)

function getindex!(iht::IHT, obj::Array{Int64, 1}, readonly=false)
    d = iht.dictionary
    if obj in keys(d)
        return d[obj]::Int64
    elseif readonly
        return -1
    end
    iht_size = capacity(iht)
    iht_count = count(iht)
    if count(iht) >= capacity(iht)
        if iht.overfullCount==0
            println("IHT full, starting to allow collisions")
        end
        iht.overfullCount += 1
        return (hash(obj) % capacity(iht))::Int64
    end
    d[obj] = iht_count
    return iht_count
end

hashcoords!(coordinates, m, readonly=false) = nothing
hashcoords!(coordinates, m::IHT, readonly=false) = getindex!(m, coordinates, readonly)
hashcoords!(coordinates, m::Integer, readonly=false) = hash(tuple(coordinates)) % m

function tiles!(ihtORsize, numtilings, floats, ints=[], readonly=false)
    qfloats = [floor(f*numtilings) for f in floats]
    tiles = zeros(Int64, numtilings)
    for tiling = 1:numtilings
        tilingX2 = tiling*2
        coords = [tiling]::Array{Int64, 1}
        b = tiling
        for (q_idx, q) in enumerate(qfloats)
            append!(coords, floor((q + b) / numtilings))
            b += tilingX2
        end
        append!(coords, ints)
        tiles[tiling] = hashcoords!(coords, ihtORsize, readonly)
    end
    return tiles
end

function tileswrap!(ihtORsize, numtilings, floats, wrapwidths, ints=[], readonly=false)
    qfloats = [floor(f*numtilings) for f in floats]
    tiles = zeros(Int64, numtilings)
    for tiling = 1:numtilings
        tilingX2 = tiling*2
        coords = [tiling]::Array{Int64, 1}
        b = tiling
        for (q_idx, q) in enumerate(qfloats)
            width = nothing
            if length(wrapwidths) >= q_idx
                width = wrapwidths[q_idx]
            end
            c = floor((q + b%numtilings) / numtilings)
            append!(coords, width==nothing ? c : c%width)
            b += tilingX2
        end
        append!(coords, ints)
        tiles[tiling] = hashcoords!(coords, ihtORsize, readonly)
    end
    return tiles
end

end

# using MimimalRLCore
"""
    TileCoder(num_tilings, num_tiles, num_features, num_ints)
Tile coder for coding all features together.
"""
mutable struct TileCoder <: AbstractFeatureConstructor
    # Main Arguments
    tilings::Int64
    tiles::Int64
    dims::Int64
    ints::Int64

    # Optional Arguments
    wrap::Bool
    wrapwidths::Float64

    iht::TileCoding.IHT
    TileCoder(num_tilings, num_tiles, num_features, num_ints=1; wrap=false, wrapwidths=0.0) =
        new(num_tilings, num_tiles, num_features, num_ints, wrap, 0.0, TileCoding.IHT(num_tilings*(num_tiles+1)^num_features * num_ints))
end

function MinimalRLCore.create_features(fc::TileCoder, s; ints=[], readonly=false)
    if fc.wrap
        return 1 .+ TileCoding.tileswrap!(fc.iht, fc.tilings, s.*fc.tiles, fc.wrapwidths, ints, readonly)
    else
        return 1 .+ TileCoding.tiles!(fc.iht, fc.tilings, s.*fc.tiles, ints, readonly)
    end
end

MinimalRLCore.feature_size(fc::TileCoder) = fc.tilings*(fc.tiles+1)^fc.dims * fc.ints
Base.size(fc::TileCoder) = fc.tilings*(fc.tiles+1)^fc.dims * fc.ints

(fc::TileCoder)(s; ints=[], readonly=false) =
    MinimalRLCore.create_features(fc, s; ints=ints, readonly=readonly)
