module TimeSeries

import DataStructures: CircularBuffer

export MSO, SineWave, MackeyGlass,
    init!, start!, step!

abstract type TimeSeries end

init!(self::TimeSeries) = nothing
step!(self::TimeSeries, action) = step!(self::TimeSeries)

mutable struct MSO <: TimeSeries
    dataset::Vector{Float64}
    idx::Int

    state::Vector{Float64}

    function MSO(max_steps::Int)
        Ω = [0.2, 0.311, 0.42, 0.51]

        values = map(θ -> sum([sin(θ*ω) for ω in Ω]), 0:max_steps)
        return new(values, 1, [0.0])
    end
end

function start!(self::MSO)
    self.idx = 1
    return step!(self)[2]
end

function step!(self::MSO)
    self.state[1] = self.dataset[self.idx]
    self.idx += 1
    return self.state[1], self.state, false
end

mutable struct SineWave <: TimeSeries
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
    return step!(self)[2]
end

function step!(self::SineWave)
    self.state[1] = self.dataset[self.idx]
    self.idx += 1
    return self.state[1], self.state, false
end

mutable struct MackeyGlass <: TimeSeries
    delta::Int
    tau::Int
    series::Float64
    history_len::Int
    history::CircularBuffer{Float64}

    state::Vector{Float64}

    function MackeyGlass(delta=10, tau=17, series=1.2)
        history_len = delta*tau
        history = CircularBuffer{Float64}(history_len)
        fill!(history, 0.0)
        return new(delta, tau, series, history_len, history, [0.0])
    end
end

function start!(self::MackeyGlass)
    return step!(self)[2]
end

function step!(self::MackeyGlass)
    for _ in 1:self.delta
        xtau = self.history[1]
        push!(self.history, self.series)
        h = self.history[end]
        self.series = h + ((0.2xtau)/(1.0+xtau^10) - 0.1h) / self.delta
    end
    self.state[1] = self.series
    return self.series, self.state, false
end

end # module TimeSeries
