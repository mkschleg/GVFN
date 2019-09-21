module CritterbotUtils

using HDF5: h5open

# ====================
# --- Random utils ---
# ====================

# const DOWN_LOC = "https://drive.google.com/open?id=1NJiwllYgibeEj2CJCDO9UnPqqZr0c-3J"
const BASE_DIR = joinpath(@__DIR__,"../../critterbot_data/")
dataFilename() = joinpath(BASE_DIR, "critterbot_data.h5")
splitStrip(st::String) = split(strip(st, [' ','\n']), " ")

# according to adam
relevant_sensors() = [2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,
                    20,21,22,26,27,28,29,30,31,32,33,34,35,36,37,
                    38,39,40,41,42,43,76,77,78,79,80,81,82,83,84,
                    85,86,88,89,90].+1
numSensors() = length(relevant_sensors())

# ========================
# --- getting the data ---
# ========================

getData() = renameData(_getData())

function getData(label::String)
    data = nothing
    h5open(dataFilename()) do f
        data = read(f[label])
    end
    return data
end

getData(toplabel::String, label::String) = getData()[toplabel][label]

# =============
# --- utils ---
# =============

function _getData()
    if !isfile(dataFilename())
        makeData()
    end

    tmpData = Dict()
    h5open(dataFilename()) do f
        datakeys = names(f)
        for k in datakeys tmpData[k] = read(f[k]) end
    end
    return tmpData
end

function renameData(tmpData::Dict)
    data=Dict()
    data["Targets"] = Dict()
    data["SensorIndices"] = Dict()
    for (k,v) in tmpData
        if endswith(k,"Target")
            data["Targets"][replace(k,"Target"=>"")] = tmpData[k]
        elseif endswith(k, "Idx")
            data["SensorIndices"][tmpData[k]] = replace(k,"Idx"=>"")
        else
            data[k] = v
        end
    end
    return data
end

function getIndexMapping()
    return getData()["SensorIndices"]
end

function getLabelMapping()
    indexmap = getIndexMapping()
    d=Dict()
    for (k,v) in indexmap
        d[v]=k
    end
    return d
end

function getReturns(data::Dict, γ::Float64)
    returns = Dict()
    for (k,v) in data
        rs = zeros(Float64, length(v))
        for i=length(rs)-1:-1:1
            rs[i] = v[i+1] + γ*rs[i+1]
        end
        returns[k] = rs
    end
    return returns
end

getReturns(γ::Float64) = getReturns(getData()["Targets"], γ)

# ====================================
# --- Make the data from raw files ---
# ====================================

function makeFeatures()
    # Features from the multi-timescale nexting paper
    tiles = Vector{Int}[]
    f = open(joinpath(BASE_DIR, "tiled_sensors.txt"))
    for l in eachline(f)
        t = map(t->parse(Int, t)+1, splitStrip(l))
        push!(tiles, t)
    end

    feats = zeros(Int, length(tiles[1]), length(tiles))
    for i=1:length(tiles)
        feats[:,i] = tiles[i]
    end
    return feats
end

function makeTargets()
    f = open(joinpath(BASE_DIR, "plooping8-200.crtrlog"))
    relevant = relevant_sensors()
    labels = map(l->string(l), splitStrip(readline(f))[relevant])

    data = Dict()
    for l in labels
        data[l] = Float64[]
    end

    data_indices = Dict()
    for i=1:length(labels)
        data_indices[labels[i]] = i
    end

    for l in eachline(f)
        values = map( v->parse(Float64, v), splitStrip(l)[relevant])
        for i=1:length(values)
            push!(data[labels[i]],values[i])
        end
    end
    return data, data_indices
end


function makeData()
    features = makeFeatures()
    targets,indices = makeTargets()

    data = Dict()
    for (k,v) in targets data[string(k,"Target")] = v end
    for (k,v) in indices data[string(k,"Idx")] = v end
    data["Features"] = features

    h5open(dataFilename(), "w") do f
        for (k,v) in data write(f,k,v) end
    end
end

# download_data() = download(DOWN_LOC, BASE_DIR)

end # module
