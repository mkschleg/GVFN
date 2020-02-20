
using Random
using MinimalRLCore


module CompassWorldConst
# Actions
const LEFT = 1
const RIGHT = 2
const FORWARD = 3

# Directions
const NORTH = 0
const EAST = 1
const SOUTH = 2
const WEST = 3

const ORANGE = 1
const YELLOW = 2
const RED = 3
const BLUE = 4
const GREEN = 5
const WHITE = 6

const NUM_COLORS = 6

const DIR_CHAR = Dict([NORTH=>'^', SOUTH=>'v', EAST=>'>', WEST=>'<'])

end

"""
 CompassWorld

   width=8, height=8
     |----width----|
   g o o o o o o o o y
   g w w w w w w w w y -
   b w w w w w w w w y h
   b w w w w w w w w y e
   b w w w w w w w w y i
   b w w w w w w w w y g
   b w w w w w w w w y h
   b w w w w w w w w y t
   b w w w w w w w w y -
   b r r r r r r r r y

"""
mutable struct CompassWorld <: AbstractEnvironment
    world_dims::NamedTuple{(:width, :height),
                           Tuple{Int, Int}}
    agent_state::NamedTuple{(:x, :y, :dir),
                            Tuple{Int, Int, Int}}
    partially_observable::Bool
    CompassWorld(width, height;
                 rng=Random.GLOBAL_RNG,
                 partially_observable=true) =
                     new((width=width, height=height),
                         (x=rand(rng, 1:width), y=rand(rng, 1:height), dir=rand(rng, 0:3)),
                         partially_observable)
end

CompassWorld(_size; kwargs...) =
    CompassWorld(_size, _size; kwargs...)

function env_settings!(as::Reproduce.ArgParseSettings,
                       env_type::Type{CompassWorld})
    Reproduce.@add_arg_table! as begin
        "--size"
        help="The length of the ring world chain"
        arg_type=Int
        default=8
    end
end



function MinimalRLCore.reset!(env::CompassWorld, rng=Random.GLOBAL_RNG)
    env.agent_state = (x=rand(rng, 1:env.world_dims.width),
                       y=rand(rng, 1:env.world_dims.height),
                       dir=rand(rng, 0:3))
end

MinimalRLCore.get_actions(env::CompassWorld) = [1, 2, 3]

get_num_features(env::CompassWorld) =
    env.partially_observable ? 6 : 3

function MinimalRLCore.environment_step!(env::CompassWorld, action::Int, rng = Random.GLOBAL_RNG)

    @boundscheck action in get_actions(env)
    
    CWC = CompassWorldConst
    x = env.agent_state.x
    y = env.agent_state.y
    dir = env.agent_state.dir
    if action == CWC.LEFT
        # Turn Left Action
        dir = (dir + 3)%4
    elseif action == CWC.RIGHT
        # Turn Right Action
        dir = (dir + 1)%4
    else
        # Forward Action
        if dir == CWC.NORTH
            y = clamp(y-1, 1, env.world_dims.height)
        elseif dir == CWC.SOUTH
            y = clamp(y+1, 1, env.world_dims.height)
        elseif dir == CWC.WEST
            x = clamp(x-1, 1, env.world_dims.width)
        elseif dir == CWC.EAST
            x = clamp(x+1, 1, env.world_dims.width)
        end
    end

    env.agent_state = (x=x, y=y, dir=dir)
end


function MinimalRLCore.get_reward(env::CompassWorld) # -> get the reward of the environment
    return 0.0f0
end


function MinimalRLCore.get_state(env::CompassWorld) # -> get state of agent
    if env.partially_observable
        return partially_observable_state(env)
    else
        return fully_observable_state(env)
    end
end


function fully_observable_state(env::CompassWorld)
    return [env.agent_state.x,
            env.agent_state.y,
            env.agent_state.dir]
end


function partially_observable_state(env::CompassWorld)
    CWC = CompassWorldConst

    dir = env.agent_state.dir
    state = zeros(6)
    if dir == CWC.NORTH && env.agent_state.y == 1
        state[CWC.ORANGE] = 1
    elseif dir == CWC.SOUTH && env.agent_state.y == env.world_dims.height
        state[CWC.RED] = 1
    elseif dir == CWC.WEST && env.agent_state.x == 1
        if env.agent_state.y == 1
            state[CWC.GREEN] = 1
        else
            state[CWC.BLUE] = 1
        end
    elseif dir == CWC.EAST && env.agent_state.x == env.world_dims.width
        state[CWC.YELLOW] = 1
    end
    if sum(state) == 0.0
        state[CWC.WHITE] = 1
    end
    return state
end


function MinimalRLCore.is_terminal(env::CompassWorld) # -> determines if the agent_state is terminal
    return false
end


function build_gridworld_char_rep(env::CompassWorld)
    CWC = CompassWorldConst
    model = fill("W",
                 (env.world_dims.height+2,
                  env.world_dims.width+2))

    model[1,:] .= "O"
    model[end, :] .= "R"

    model[:,1] .= "B"
    model[1:2, 1] .= "G"
    model[:, end] .= "Y"

    model[env.agent_state.y+1, env.agent_state.x+1] =
        Base.string(CWC.DIR_CHAR[env.agent_state.dir])

    return model
end


function Base.string(env::CompassWorld)
    model = build_gridworld_char_rep(env)
    str = ""
    for r in 1:size(model)[1]
        str *= join(model[r,:], " ")*"\n"
    end
    return str
end


function Base.print(io::IO, env::CompassWorld)
    print(io, env.agent_state)
    print(io, "\n")
    model = build_gridworld_char_rep(env)
    println(size(model))
    for r in 1:size(model)[1]
        print(io, join(model[r,:], " ")*"\n")
    end
end

