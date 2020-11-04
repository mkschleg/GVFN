module CritterbotUtils

using HDF5: h5open

# const DOWN_LOC = "https://drive.google.com/open?id=1NJiwllYgibeEj2CJCDO9UnPqqZr0c-3J"
const BASE_DIR = joinpath(@__DIR__,"../../raw_data/Critterbot/")

numSteps() = 119000
numFeatures() = 8197

function getData(filename::String)
    data = nothing
    h5open(filename) do f
        data = read(f["data"])
    end
    return data
end

# according to adam
relevant_sensors() = getData(joinpath(BASE_DIR, "relevantSensors.h5")) .+ 1 # NOTE: +1 for 1-based indexing
numRelevantSensors() = length(relevant_sensors())

loadTiles() = getData(joinpath(BASE_DIR, "tiles.h5")) .+ 1 # NOTE: +1 to get 1-based indexing
loadSensor(sensorIdx::Int) = getData(joinpath(BASE_DIR,"sensors/sensor$(sensorIdx).h5"))

# TODO: eventually when a subset of the sensors is selected we should save that data in 1 file
# so that we can load the subset with only 1 disk read
loadSensor(indices::Vector{Int}) = hcat([loadSensor(idx) for idx ∈ indices])


function getSensorName(sensorIdx)
    names = getData(joinpath(BASE_DIR, "sensorNames.h5"))
    return names[sensorIx]
end

function getSensorIndex(sensorName)
    names = getData(joinpath(BASE_DIR, "sensorNames.h5"))
    for i=1:length(names)
        if names[i] == sensorName
            return i
        end
    end
    error("Sensor $(sensorName) not found")
end

function getReturns(indices::Vector{Int}, γ::Float64)
    sensors = loadSensor(indices)
    returns = zeros(numSteps(), length(indices))
    for i=length(rs)-1:-1:1
        returns[:,i] = sensors[:,i+1] + γ*returns[:,i+1]
    end
    return returns
end

end # module
