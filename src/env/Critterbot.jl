# import CritterbotUtils

# include("utils/critterbot.jl")

mutable struct Critterbot{T<:Number} <: AbstractEnvironment
    nfeatures::Int
    maxSteps::Int
    step::Int

    ϕ::Matrix{T}
    targets::Vector{Float64}
end

# tilecoding from the original paper
# sensor_name: e.g. Mag0, Light3, IRLight0
function OriginalTiledCritterbot(sensor_name::String, max_steps::Int=119000)

    memsize = 8192

    data = CritterbotUtils.getData()
    targets = data["Targets"][sensor_name]
    feats = data["Features"]

    Φ = zeros(Float64, memsize, max_steps)
    for i=1:max_steps
        Φ[:,feats[i]].= 1.0
    end

    return Critterbot(memsize, max_steps, 1, Φ, targets)
end

# Uses a collection of raw sensor readings as a feature vectors
function RawCritterbot(targetSensor::String, featureSensors::Vector{String}, max_steps::Int=119000)

    data = CritterbotUtils.getData()

    targets = data["Targets"][targetSensor]
    Φ = transpose(hcat(map(s->data["Targets"][s], featureSensors)...))
    nfeats = size(Φ,1)

    return Critterbot(nfeats, max_steps, 1, copy(Φ), targets)
end

function RawCritterbot(targetSensor::Int, featureSensors::Vector{Int}, max_steps::Int = 119000)
    data = CritterbotUtils.getData()

    targetSensor = data["SensorIndices"][targetSensor]
    sensors = map(s->data["SensorIndices"][s], featureSensors)
    return RawCritterbot(targetSensor, sensors, max_steps)
end

nfeatures(self::Critterbot) = self.nfeatures

start!(self::Critterbot) = self.ϕ[:,1]

function step!(self::Critterbot)
    # return float, float[], bool
    self.step+=1
    terminal = step == self.maxSteps ? true : false
    return self.targets[self.step], self.ϕ[:, self.step], terminal
end

JuliaRL.reset!(env::Critterbot; rng = Random.GLOBAL_RNG, kwargs...) = start(env)

JuliaRL.environment_step!(env::Critterbot,
                          action;
                          rng = Random.GLOBAL_RNG,
                          kwargs...) = step!(env)

JuliaRL.get_actions(env::Critterbot) = Set()
