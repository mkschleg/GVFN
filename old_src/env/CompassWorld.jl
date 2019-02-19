module CompassWorld

using Random
import Base.+
import Base.-

@enum Directions begin
    NORTH = 1
    EAST
    SOUTH
    WEST
end

+(dir::Directions, a::Int64) = Directions( ((Int(dir) - 1 + a) % 4) + 1 )
-(dir::Directions, a::Int64) = Directions( ((((Int(dir) - 1 - a) % 4) + 4) % 4) + 1 )

@enum Colors begin
    ORANGE = 1
    YELLOW
    RED
    BLUE
    GREEN
    WHITE
end

@enum ACTIONS begin
    FORWARD = 1
    LEFT
    RIGHT
end

num_actions = 3
action_space() = 1:3

mutable struct State
    x
    y
end

mutable struct Environment
    pos
    dir
    size
    Environment(size; rng=Random.GLOBAL_RNG) = new(State(rand(rng, 1:size), rand(rng, 1:size)), NORTH, size)
end

function get_observation(env::Environment)
    obs = zeros(6)
    size = env.size
    if env.dir == NORTH
        if env.pos.y == 1
            obs[Int(ORANGE)] = 1.0
        end
    elseif env.dir == SOUTH
        if env.pos.y == size
            obs[Int(RED)] = 1.0
        end
    elseif env.dir == EAST
        #Yellow Wall
        if env.pos.x == size
            obs[Int(YELLOW)] = 1.0
        end
    else
        # West Blue Green Wall
        if env.pos.x == 1
            if env.pos.y == 1
                obs[Int(GREEN)] = 1.0
            else
                obs[Int(BLUE)] = 1.0
            end
        end
    end

    if sum(obs) == 0
        # White observation
        obs[end] = 1
    end

    return obs
end

init(rng=Random.GLOBAL_RNG) = Environment(8; rng=rng)

function start!(env::Environment; rng=Random.GLOBAL_RNG)
    env.pos = (x = rand(rng, 1:size), y = rand(rng, 1:size))
    env.dir = NORTH
    return env
end

function move_forward!(env_state)
    if env_state.dir == NORTH
        env_state.pos.y = clamp(env_state.pos.y - 1, 1, env_state.size)
    elseif env_state.dir == SOUTH
        env_state.pos.y = clamp(env_state.pos.y + 1, 1, env_state.size)
    elseif env_state.dir == WEST
        env_state.pos.x = clamp(env_state.pos.x - 1, 1, env_state.size)
    elseif env_state.dir == EAST
        env_state.pos.x = clamp(env_state.pos.x + 1, 1, env_state.size)
    else
        throw(ErrorException("Direction went wrong..."))
    end
end

function step!(env_state, action; rng=Random.GLOBAL_RNG)

    if action == Int(FORWARD)
        move_forward!(env_state)
    elseif action == Int(LEFT)
        env_state.dir -= 1
    elseif action == Int(RIGHT)
        env_state.dir += 1
    else
        throw(ErrorException("Action not in range..."))
    end

    return env_state, 0, false
end

end
