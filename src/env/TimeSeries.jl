import DataStructures: CircularBuffer

abstract type TimeSeriesEnv <: AbstractEnvironment end

init!(self::TimeSeriesEnv) = nothing
JuliaRL.environment_step!(self::ENV, action; rng = Random.GLOBAL_RNG, kwargs...) where {ENV<:TimeSeriesEnv} =
    _step!(self)
JuliaRL.reset!(self::ENV; rng=Random.GLOBAL_RNG, kwargs...) where {ENV<:TimeSeriesEnv} =
    _start!(self)

get_num_features(self::TimeSeriesEnv) = 1

JuliaRL.get_actions(self::TimeSeriesEnv) = Set(1)
JuliaRL.get_reward(self::TimeSeriesEnv) = 0



mutable struct MSO <: TimeSeriesEnv
    dataset::Vector{Float64}
    idx::Int

    state::Vector{Float64}

    function MSO(max_steps::Int)
        Ω = [0.2, 0.311, 0.42, 0.51]

        values = map(θ -> sum([sin(θ*ω) for ω in Ω]), 0:max_steps)
        return new(values, 1, [0.0])
    end
end

function _start!(self::MSO)
    self.idx = 1
    # return _step!(self)
end

function _step!(self::MSO)
    # self.state[1] = self.dataset[self.idx]
    self.idx += 1
    # return self.state
end

function JuliaRL.get_state(self::MSO) # -> get state of agent
    # env.state[env.idx]
    [self.dataset[self.idx]]
end

JuliaRL.is_terminal(self::MSO) = self.idx == length(self.dataset)

mutable struct SineWave <: TimeSeriesEnv
    dataset::Vector{Float64}
    idx::Int

    state::Vector{Float64}

    function SineWave(max_steps::Int)
        values = map(θ->sin(θ * 2π / 100), 0:max_steps)
        return new(values, 1, [0.0])
    end
end

function _start!(self::SineWave)
    self.idx = 1
    # return _step!(self)
end

function _step!(self::SineWave)
    # self.state[1] = self.dataset[self.idx]
    self.idx += 1
    # return self.state
end

function JuliaRL.get_state(self::SineWave) # -> get state of agent
    # env.state[env.idx]
    [self.dataset[self.idx]]
end

JuliaRL.is_terminal(self::SineWave) = self.idx == length(self.dataset)

mutable struct MackeyGlass <: TimeSeriesEnv
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

function _start!(self::MackeyGlass)
    _step!(self)
end

function _step!(self::MackeyGlass)
    for _ in 1:self.delta
        xtau = self.history[1]
        push!(self.history, self.series)
        h = self.history[end]
        self.series = h + ((0.2xtau)/(1.0+xtau^10) - 0.1h) / self.delta
    end
    self.state[1] = self.series
    return self.state
end

function JuliaRL.get_state(self::MackeyGlass) # -> get state of agent
    [self.series]
end

JuliaRL.is_terminal(self::MackeyGlass) = false
