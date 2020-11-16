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

mutable struct Critterbot <: TimeSeriesEnv
    num_steps::Int
    num_features::Int
    sensors::Vector{Int}

    idx::Int
    data::Array{Float64}
end

function Critterbot(obs_sensors, target_sensors)
    all_sensors = vcat(obs_sensors, target_sensors)
    num_features = length(obs_sensors)
    return Critterbot(CritterbotUtils.numSteps(),
                      num_features, all_sensors,
                      0, CritterbotUtils.loadSensor(all_sensors))
end

# Hack to use same features as targets; just duplicate the data in new cols
Critterbot(sensors::Vector{Int}) = Critterbot(sensors, sensors)
get_num_features(cb::Critterbot) = cb.num_features
get_num_targets(cb::Critterbot) = length(cb.sensors)-cb.num_features

function MinimalRLCore.start!(cb::Critterbot)
    cb.idx = 1
    return MinimalRLCore.get_state(cb)
end

function MinimalRLCore.step!(cb::Critterbot)
    cb.idx += 1
    return MinimalRLCore.get_state(cb)
end

# Data for each sensor in a row, so that we can access data for all sensors by col
MinimalRLCore.get_state(cb::Critterbot) = cb.data[1:cb.num_features, cb.idx]
MinimalRLCore.get_reward(cb::Critterbot) = cb.data[cb.num_features+1:end, cb.idx] #

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

