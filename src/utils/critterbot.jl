module CritterbotUtils

using HDF5: h5open

# const DOWN_LOC = "https://drive.google.com/open?id=1NJiwllYgibeEj2CJCDO9UnPqqZr0c-3J"
const BASE_DIR = joinpath(@__DIR__,"../../raw_data/Critterbot/")

numSteps() = 119000
numFeatures() = 8197

function getData(filename::String)
    data = nothing
    if !isfile(filename)
        throw(ErrorException("$(filename) does not exist."))
    end
    h5open(filename) do f
        data = read(f["data"])
    end
    return data
end

# according to adam
relevant_sensors() = getData(joinpath(BASE_DIR, "relevantSensors.h5")) .+ 1 # NOTE: +1 for 1-based indexing
relevant_sensor_idx() = getData(joinpath(BASE_DIR, "relevantSensorIndices.h5")) .+ 1
numRelevantSensors() = length(relevant_sensors())

loadTiles() = getData(joinpath(BASE_DIR, "tiles.h5")) .+ 1 # NOTE: +1 to get 1-based indexing
loadSensor(sensorIdx::Int) = getData(joinpath(BASE_DIR,"sensors/sensor$(sensorIdx).h5"))
loadSensor(sensorName::String) =
    getData(joinpath(BASE_DIR,"sensors/sensor$(getSensorIndex(sensorName)).h5"))

# TODO: eventually when a subset of the sensors is selected we should save that data in 1 file
# so that we can load the subset with only 1 disk read
# loadSensor(indices::Vector{Int}) = vcat([loadSensor(idx)' for idx ∈ indices]...)
loadSensor(indices::AbstractArray)  = vcat([loadSensor(idx)' for idx ∈ indices]...)
loadSensor_cmajor(indices::AbstractArray)  = hcat([loadSensor(idx) for idx ∈ indices]...)

getSensorNames() = getData(joinpath(BASE_DIR, "sensorNames.h5"))
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

function getReturns(indices::AbstractArray, γ::Float64)
    sensors = loadSensor(indices)
    returns = zero(sensors)
    for i ∈ (size(sensors)[2]-1):-1:1
        returns[:, i] = sensors[:, i+1] + γ*returns[:,i+1]
    end
    return returns
end

end # module
