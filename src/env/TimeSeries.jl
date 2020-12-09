using MinimalRLCore

import DataStructures: CircularBuffer
import HDF5: h5read

import .CritterbotUtils

"""
    TimeSeriesEnv


Basic abstract type for timeseries data sets.

"""
abstract type TimeSeriesEnv <: AbstractEnvironment end

init!(self::TimeSeriesEnv) = nothing
MinimalRLCore.step!(self::TimeSeriesEnv, action) = MinimalRLCore.step!(self::TimeSeriesEnv)

get_num_features(self::TimeSeriesEnv) = 1
get_num_targets(self::TimeSeriesEnv) = 1

# ==================
# --- CRITTERBOT ---
# ==================

squish(vec) = (vec .- minimum(vec; dims=2)) ./ (maximum(vec; dims=2) - minimum(vec; dims=2))

mutable struct Critterbot <: TimeSeriesEnv
    num_steps::Int
    num_features::Int
    num_targets::Int
    sensors::Vector{Int}

    idx::Int
    data::Array{Float64}
end

function Critterbot(obs_sensors, target_sensors)
    all_sensors = vcat(obs_sensors, target_sensors)
    num_features = length(obs_sensors)
    return Critterbot(CritterbotUtils.numSteps(),
                      num_features, length(target_sensors), CritterbotUtils.getSensorIndices(all_sensors),
                      0, squish(CritterbotUtils.loadSensor(all_sensors)))
end

function Critterbot(obs_sensors, target_sensors, γs::AbstractArray)
    all_sensors = vcat(obs_sensors, target_sensors)
    num_features = length(obs_sensors)*length(γs)
    data = squish(vcat(CritterbotUtils.getReturns(obs_sensors, γs), CritterbotUtils.loadSensor(target_sensors)))
    
    return Critterbot(CritterbotUtils.numSteps(),
                      num_features, length(target_sensors), CritterbotUtils.getSensorIndices(all_sensors),
                      0, data)
end

Critterbot(obs_sensors, target_sensors, γ_str::String) =
    Critterbot(obs_sensors, target_sensors, eval(Meta.parse(γ_str)))

# Hack to use same features as targets; just duplicate the data in new cols
Critterbot(sensors::Vector{Int}) = Critterbot(sensors, sensors)
get_num_features(cb::Critterbot) = cb.num_features
get_num_targets(cb::Critterbot) = cb.num_targets

function MinimalRLCore.start!(cb::Critterbot)
    cb.idx = 1
    return MinimalRLCore.get_state(cb)
end

function MinimalRLCore.step!(cb::Critterbot)
    cb.idx += 1
    return MinimalRLCore.get_state(cb), MinimalRLCore.get_reward(cb)
end

# Data for each sensor in a row, so that we can access data for all sensors by col
MinimalRLCore.get_state(cb::Critterbot) = cb.data[1:cb.num_features, cb.idx]
MinimalRLCore.get_reward(cb::Critterbot) = cb.data[cb.num_features+1:end, cb.idx] #


mutable struct CritterbotTPC <: TimeSeriesEnv
    num_steps::Int
    num_features::Int

    idx::Int
    obs_data::Array{Float64}
    rewards::Array{Float64}
    discounts::Array{Float64}
    the_all_seeing_eye::Array{Float64}
end

function CritterbotTPC(obs_sensors; γ=0.9875)
    # all_sensors = vcat(obs_sensors, target_sensors)

    volts = CritterbotUtils.loadSensor(["Motor$(i)0" for i in 0:2])
    cur = CritterbotUtils.loadSensor(["Motor$(i)2" for i in 0:2])

    rewards = sum(abs.(cur .* volts); dims=1)
    light3 = GVFN.CritterbotUtils.loadSensor("Light3")
    discounts = (light3 .< 1020) .* γ

    num_features = length(obs_sensors)
    feats = squish(CritterbotUtils.loadSensor(obs_sensors))

    the_all_seeing_eye = CritterbotUtils.loadSensor("powerToGoal-$(γ)")
    
    return CritterbotTPC(
        CritterbotUtils.numSteps(),
        num_features,        
        0,
        feats,
        rewards,
        discounts,
        the_all_seeing_eye)
end

function CritterbotTPC(obs_sensors, γs::AbstractArray; γ=0.9875)
    # all_sensors = vcat(obs_sensors, target_sensors)
    num_features = length(obs_sensors)*length(γs)
    feats = squish(CritterbotUtils.getReturns(obs_sensors, γs))

    volts = CritterbotUtils.loadSensor(["Motor$(i)0" for i in 0:2])
    cur = CritterbotUtils.loadSensor(["Motor$(i)2" for i in 0:2])

    rewards = sum(abs.(cur .* volts); dims=1)
    light3 = GVFN.CritterbotUtils.loadSensor("Light3")
    discounts = (light3 .< 1020).*γ

    the_all_seeing_eye = CritterbotUtils.loadSensor("powerToGoal-$(γ)")
    
    return CritterbotTPC(
        CritterbotUtils.numSteps(),
        num_features,        
        0,
        feats,
        rewards,
        discounts,
        the_all_seeing_eye)
end

CritterbotTPC(obs_sensors, γ_str::String) =
    CritterbotTPC(obs_sensors, eval(Meta.parse(γ_str)))

# Hack to use same features as targets; just duplicate the data in new cols
# CritterbotTPC(sensors::Vector{Int}) = Critterbot(sensors, sensors)
get_num_features(cb::CritterbotTPC) = cb.num_features
# get_num_targets(cb::CritterbotTPC) = 1

function MinimalRLCore.start!(cb::CritterbotTPC)
    cb.idx = 1
    return MinimalRLCore.get_state(cb)
end

function MinimalRLCore.step!(cb::CritterbotTPC)
    cb.idx += 1
    return MinimalRLCore.get_state(cb), MinimalRLCore.get_reward(cb)
end

# Data for each sensor in a row, so that we can access data for all sensors by col
MinimalRLCore.get_state(cb::CritterbotTPC) = cb.obs_data[:, cb.idx]
MinimalRLCore.get_reward(cb::CritterbotTPC) = [cb.rewards[cb.idx], cb.discounts[cb.idx]]
ground_truth(cb::CritterbotTPC) = [cb.the_all_seeing_eye[cb.idx]]


# ===========
# --- MSO ---
# ===========
"""
    MSO

Multiple Superimposed Oscillator:
  y(t) = sin(0.2t) + sin(0.311t) + sin(0.42t) + sin(0.51t)

"""
mutable struct MSO <: TimeSeriesEnv
    θ::Int
    Ω::Vector{Float64}

    state::Vector{Float64}
end

MSO() = MSO(1, [0.2, 0.311, 0.42, 0.51], [0.0])

function MinimalRLCore.start!(self::MSO)
    self.θ = 1
    return step!(self)
end

function MinimalRLCore.step!(self::MSO)
    self.state[1] = sum([sin(self.θ*ω) for ω in self.Ω])
    self.θ += 1
    return self.state
end

MinimalRLCore.get_state(self::MSO) = self.state

# =================
# --- SINE WAVE ---
# =================
"""
    SineWave

Simple sine wave dataset for debugging.
"""
mutable struct SineWave <: TimeSeriesEnv
    dataset::Vector{Float64}
    idx::Int

    state::Vector{Float64}

    function SineWave(max_steps::Int)
        values = map(θ->sin(θ * 2π / 100), 0:max_steps)
        return new(values, 1, [0.0])
    end
end

function MinimalRLCore.start!(self::SineWave)
    self.idx = 1
    return step!(self)
end

function MinimalRLCore.step!(self::SineWave)
    self.state[1] = self.dataset[self.idx]
    self.idx += 1
    return self.state
end

# ===================
# --- MACKEYGLASS ---
# ===================
"""
    MackeyGlass

Mackey-Glass synthetic dataset
"""
mutable struct MackeyGlass <: TimeSeriesEnv
    delta::Int
    tau::Int
    series::Float64
    history_len::Int
    history::CircularBuffer{Float64}

    state::Vector{Float64}
end

function MackeyGlass(delta=10, tau=17, series=1.2)
    history_len = delta*tau
    history = CircularBuffer{Float64}(history_len)
    fill!(history, 0.0)
    return MackeyGlass(delta, tau, series, history_len, history, [0.0])
end

function MinimalRLCore.start!(self::MackeyGlass)
    return step!(self)
end

function MinimalRLCore.step!(self::MackeyGlass)
    for _ in 1:self.delta
        xtau = self.history[1]
        push!(self.history, self.series)
        h = self.history[end]
        self.series = h + ((0.2xtau)/(1.0+xtau^10) - 0.1h) / self.delta
    end
    self.state[1] = self.series
    return self.state
end

# ============
# --- ACEA ---
# ============
"""
    ACEA
"""
mutable struct ACEA <: TimeSeriesEnv
    data::Vector{Float64}
    idx::Int

    state::Vector{Float64}

end

function ACEA()
    return ACEA(
        h5read(joinpath(@__DIR__, "../../raw_data/acea.h5"), "data"),
        1,
        [0.0]
    )
end

function MinimalRLCore.start!(self::ACEA)
    self.idx = 1
    return step!(self)
end

function MinimalRLCore.step!(self::ACEA)
    obs = self.data[self.idx]
    self.idx+=1
    self.state[1] = obs
    return self.state
end

