import DataStructures: CircularBuffer
import HDF5: h5read

abstract type TimeSeriesEnv end

init!(self::TimeSeriesEnv) = nothing
step!(self::TimeSeriesEnv, action) = step!(self::TimeSeriesEnv)

get_num_features(self::TimeSeriesEnv) = 1

# ===========
# --- MSO ---
# ===========

mutable struct MSO <: TimeSeriesEnv
    θ::Int
    Ω::Vector{Float64}

    state::Vector{Float64}
end

MSO() = MSO(1, [0.2, 0.311, 0.42, 0.51], [0.0])

function start!(self::MSO)
    self.θ = 1
    return step!(self)
end

function step!(self::MSO)
    self.state[1] = sum([sin(self.θ*ω) for ω in self.Ω])
    self.θ += 1
    return self.state
end

# =================
# --- SINE WAVE ---
# =================

mutable struct SineWave <: TimeSeriesEnv
    dataset::Vector{Float64}
    idx::Int

    state::Vector{Float64}

    function SineWave(max_steps::Int)
        values = map(θ->sin(θ * 2π / 100), 0:max_steps)
        return new(values, 1, [0.0])
    end
end

function start!(self::SineWave)
    self.idx = 1
    return step!(self)
end

function step!(self::SineWave)
    self.state[1] = self.dataset[self.idx]
    self.idx += 1
    return self.state
end

# ===================
# --- MACKEYGLASS ---
# ===================

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

function start!(self::MackeyGlass)
    return step!(self)
end

function step!(self::MackeyGlass)
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

function start!(self::ACEA)
    self.idx = 1
    return step!(self)
end

function step!(self::ACEA)
    obs = self.data[self.idx]
    self.idx+=1
    self.state[1] = obs
    return self.state
end

