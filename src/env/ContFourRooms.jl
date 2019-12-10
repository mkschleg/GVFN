# module FourRooms

import Random, Base.size

# import IWER.GeneralValueFunction


module ContFourRoomsParams

BASE_WALLS = [0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 1 1 1 1 1 0;
              0 0 0 0 0 0 0 0 0 1 0;
              0 0 0 0 0 1 1 1 0 1 0;
              0 0 0 0 0 1 0 0 0 1 0;
              1 0 1 1 1 1 0 0 0 1 0;
              0 0 0 0 0 1 1 1 0 1 0;
              0 0 0 0 0 1 0 0 0 0 0;
              0 0 0 0 0 1 0 0 0 0 0;
              0 0 0 0 0 0 0 0 0 0 0;
              0 0 0 0 0 1 0 0 0 0 0;]

COORIDOR_WALLS = [0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 1 1 1 1 1 0;
                  0 0 0 0 0 0 0 0 0 1 0;
                  0 0 0 0 0 1 1 1 0 1 0;
                  0 0 0 0 0 1 0 0 0 1 0;
                  1 0 1 1 1 1 0 0 0 1 0;
                  0 0 0 0 0 1 1 1 0 1 0;
                  0 0 0 0 0 1 0 0 0 0 0;
                  0 0 0 0 0 1 0 0 0 0 0;
                  0 0 0 0 0 0 0 0 0 0 0;
                  0 0 0 0 0 1 0 0 0 0 0;]

UP = 1
RIGHT = 2
DOWN = 3
LEFT = 4

ACTIONS = [UP, RIGHT, DOWN, LEFT]

ROOM_TOP_LEFT = 1
ROOM_TOP_RIGHT = 2
ROOM_BOTTOM_LEFT = 3
ROOM_BOTTOM_RIGHT = 4


const BODY_RADIUS = 0.1
const STEP = 0.5

const AGENT_BOUNDRIES = [
    [-BODY_RADIUS, 0.0], #Top
    [0.0, -BODY_RADIUS], #Left
    [BODY_RADIUS, 0.0], #Bottom
    [0.0, BODY_RADIUS], #Right
]

end


"""
    ContFourRooms

    Four Rooms environment using JuliaRL abstract environment.

    - state: [y, x]


"""
mutable struct ContFourRooms <: AbstractEnvironment
    state::Array{Float64, 1}
    walls::Array{Bool, 2}
    max_action_noise::Float64
    drift_noise::Float64
    partial::Bool
    normalized::Bool
    collision::Bool
    edge_locs::Array{Array{Int64, 1}, 1}
    collision_check::Array{Bool, 1}
    new_state::Array{Float64, 1}
    ContFourRooms(size, walls, max_action_noise, drift_noise, normalized, partial_info) =
        new(size, walls, max_action_noise, drift_noise, partial_info, normalized, false, [[0,0] for i in 1:4], fill(false, 4), zeros(2))
end


ContFourRooms(max_action_noise = 0.1, drift_noise = 0.001; normalize=false, partial_info=false) =
    ContFourRooms([0.0,0.0], convert(Array{Bool}, ContFourRoomsParams.BASE_WALLS), max_action_noise, drift_noise, normalize, partial_info)

ContFourRooms(walls::Array{Int64, 2}, max_action_nois, drift_noise; normalize=false, partial_info=false) =
        ContFourRooms([0.0,0.0], convert(Array{Bool}, walls), max_action_noise, drift_noise, normalize, partial_info)

function ContFourRooms(size::Int, wall_list::Array{CartesianIndex{2}}, max_action_noise::Float64, drift_noise::Float64)
    walls = fill(false, size[1], size[2])
    for wall in wall_list
        walls[wall] = true
    end
    ContFourRooms([0.0,0.0], walls, max_noise, drift_noise, false)
end

JuliaRL.is_terminal(env::ContFourRooms) = false
JuliaRL.get_reward(env::ContFourRooms) = 0
function JuliaRL.get_state(env::ContFourRooms)
    if env.partial
        (Float64.(env.collision_check), env.state, env.collision)
    else
        ((env.normalized ? [env.state./size(env); Float64.(env.collision_check)] : [env.state; Float64.(env.collision_check)]), env.collision)
    end
end
project(env::ContFourRooms, state) = [Int64(floor(state[1])) + 1, Int64(floor(state[2])) + 1]
project(env::ContFourRooms, state, loc) = begin; loc[1] = Int64(floor(state[1]) + 1); loc[2] = Int64(floor(state[2]) + 1); end;
function is_wall(env::ContFourRooms, state::Array{Float64, 1})
    prj = project(env, state)
    return is_wall(env, prj)
end

function is_wall(env::ContFourRooms, prj::Array{Int64, 1})
    if prj[1] < 1 || prj[2] < 1 || prj[1] > 11 || prj[2] > 11
        return false
    end
    env.walls[prj[1], prj[2]]
end

random_state(env::ContFourRooms, rng) = [rand(rng)*size(env.walls)[1], rand(rng)*size(env.walls)[2]]
function random_start_state(env::ContFourRooms, rng)
    state = random_state(env, rng)
    while is_wall(env, state)
        state = random_state(env, rng)
    end
    return state
end
Base.size(env::ContFourRooms) = size(env.walls)
num_actions(env::ContFourRooms) = 4
get_states(env::ContFourRooms) = findall(x->x==false, env.walls)
JuliaRL.get_actions(env::ContFourRooms) = ContFourRoomsParams.ACTIONS


function handle_collision(env::ContFourRooms, state, action)

    #Approximate by the square...
    frp = ContFourRoomsParams
    new_state = copy(state)

    new_state[1] = clamp(new_state[1], frp.BODY_RADIUS, size(env.walls)[1] - frp.BODY_RADIUS)
    new_state[2] = clamp(new_state[2], frp.BODY_RADIUS, size(env.walls)[2] - frp.BODY_RADIUS)

    # Really basic collision detection for 2-d plane worlds.

    # collided = new_state[1] != state[1] || new_state[2] != state[2]

    for i in 1:4
        project(env::ContFourRooms, new_state .+ frp.AGENT_BOUNDRIES[i], env.edge_locs[i])
        env.collision_check[i] = is_wall(env, env.edge_locs[i])
    end

    # collided = collided || any(env.collision_check)
    collided = any(env.collision_check)

    if env.collision_check[1] && env.collision_check[2]
        wall_piece = project(env, new_state)
        if action == frp.UP
            new_state[1] = (wall_piece[1]) + frp.BODY_RADIUS
        elseif action == frp.DOWN
            new_state[1] = (wall_piece[1] - 1) - frp.BODY_RADIUS
        end
    elseif env.collision_check[1]
        new_state[1] = (env.edge_locs[1][1] - 1) - frp.BODY_RADIUS
    elseif env.collision_check[2]
        new_state[1] = (env.edge_locs[2][1]) + frp.BODY_RADIUS
    end

    if env.collision_check[3] && env.collision_check[4]
        wall_piece = project(env, new_state)
        if action == frp.LEFT
            new_state[2] = (wall_piece[2]) + frp.BODY_RADIUS
        elseif action == frp.RIGHT
            new_state[2] = (wall_piece[2] - 1) - frp.BODY_RADIUS
        end
    elseif env.collision_check[3]
        new_state[2] = (env.edge_locs[3][2] - 1) - frp.BODY_RADIUS
    elseif env.collision_check[4]
        new_state[2] = (env.edge_locs[4][2]) + frp.BODY_RADIUS
    end

    new_state[1] = clamp(new_state[1], frp.BODY_RADIUS, size(env.walls)[1] - frp.BODY_RADIUS)
    new_state[2] = clamp(new_state[2], frp.BODY_RADIUS, size(env.walls)[2] - frp.BODY_RADIUS)


    if new_state[1] == 0.1
        env.collision_check[1] = true
    end
    if new_state[1] == 10.9
        env.collision_check[3] = true
    end
    if new_state[2] == 0.1
        env.collision_check[2] = true
    end
    if new_state[2] == 10.9
        env.collision_check[4] = true
    end

    
    return new_state, collided
end

function handle_collision!(env::ContFourRooms, action)
    frp = ContFourRoomsParams
    env.state, collided = handle_collision(env, env.state, action)
    return collided
end


function which_room(env::ContFourRooms, state)
    frp = ContFourRoomsParams
    room = -1
    if state[1] < 6
        # LEFT
        if state[2] < 6
            # TOP
            room = frp.ROOM_TOP_LEFT
        else
            # Bottom
            room = frp.ROOM_BOTTOM_LEFT
        end
    else
        # RIGHT
        if state[2] < 7
            # TOP
            room = frp.ROOM_TOP_RIGHT
        else
            # Bottom
            room = frp.ROOM_BOTTOM_RIGHT
        end
    end
    return room
end

function JuliaRL.reset!(env::ContFourRooms; rng=Random.GLOBAL_RNG, kwargs...)
    state = random_state(env, rng)

    while is_wall(env, state)
        state = random_state(env, rng)
    end
    env.state = state
    return state
end

function mini_step!(env::ContFourRooms, step, action)

    env.state .+= step
    collision = handle_collision!(env, action)
    return collision
end

function mini_step(env::ContFourRooms, state, step, action)

    new_state = state .+ step
    new_state, collision = handle_collision(env, new_state, action)
    return new_state, collision
end

function JuliaRL.environment_step!(env::ContFourRooms, action; rng=Random.GLOBAL_RNG, kwargs...)


    frp = ContFourRoomsParams
    next_step = zeros(2)

    if action == frp.UP
        next_step[1] = -(frp.STEP - rand(rng)*env.max_action_noise)
        next_step[2] = randn(rng)*env.drift_noise
    elseif action == frp.DOWN
        next_step[1] = frp.STEP - rand(rng)*env.max_action_noise
        next_step[2] = randn(rng)*env.drift_noise
    elseif action == frp.RIGHT
        next_step[2] = frp.STEP - rand(rng)*env.max_action_noise
        next_step[1] = randn(rng)*env.drift_noise
    elseif action == frp.LEFT
        next_step[2] = -(frp.STEP - rand(rng)*env.max_action_noise)
        next_step[1] = randn(rng)*env.drift_noise
    else
        throw("Wrong action")
    end

    # mini_physics simulation for 1 second (== 10 steps of 0.1 seconds)
    Δt=1.0
    τ=10
    next_step .*= Δt/τ
    collision = false
    for t in 1:τ
        collision = mini_step!(env, next_step, action)
        if collision
            break
        end
    end
    env.collision = collision
end

function _step(env::ContFourRooms, state, action; rng=Random.GLOBAL_RNG, kwargs...)

    frp = ContFourRoomsParams
    next_step = zeros(2)

    if action == frp.UP
        next_step[1] = -(frp.STEP - rand(rng)*env.max_action_noise)
        next_step[2] = randn(rng)*env.drift_noise
    elseif action == frp.DOWN
        next_step[1] = frp.STEP - rand(rng)*env.max_action_noise
        next_step[2] = randn(rng)*env.drift_noise
    elseif action == frp.RIGHT
        next_step[2] = frp.STEP - rand(rng)*env.max_action_noise
        next_step[1] = randn(rng)*env.drift_noise
    elseif action == frp.LEFT
        next_step[2] = -(frp.STEP - rand(rng)*env.max_action_noise)
        next_step[1] = randn(rng)*env.drift_noise
    else
        throw("Wrong action")
    end

    Δt=1.0
    τ=10
    next_step .*= Δt/τ
    collision = false
    next_state = state[1]
    for t in 1:τ
        next_state, collision = mini_step(env, next_state, next_step, action)
        if collision
            break
        end
    end
    # next_state = handle_collision(env, next_state, action)
    return env.state,  JuliaRL.get_state(env), 0.0, false
end

function MonteCarloReturn(env::ContFourRooms, gvf::GVF, start_state::Array{Float64, 1},
                          num_returns::Int64, γ_thresh::Float64=1e-6,
                          max_steps::Int64=Int64(1e7);
                          rng=Random.GLOBAL_RNG)
    throw("Broken...")
    returns = zeros(num_returns)
    for ret in 1:num_returns
        step = 0
        cumulative_gamma = 1.0
        while cumulative_gamma > γ_thresh && step < max_steps
            action = StatsBase.sample(rng, gvf.policy, cur_state)
            _, next_state, _, _ = step!(env, cur_state, action; rng=rng)
            println(st)
            c, γ, pi_prob = get(gvf, st[1], action, stp1, nothing, nothing)
            returns[ret] += cumulative_gamma*c
            cumulative_gamma *= γ
            cur_state = next_state
            st = stp1
            step += 1
        end
    end

    return returns
end

import ProgressMeter

function get_sequence(env::ContFourRooms, num_steps, policy; seed=0, normalized=false)
    rng = Random.MersenneTwister(seed)
    env.normalized = normalized
    states = Array{Array{Float64, 1}}(undef, num_steps+1)
    _, state = start!(env; rng=rng)
    states[1] = copy(state[1])
    ProgressMeter.@showprogress 0.1 "Step:" for step in 1:num_steps
        action = StatsBase.sample(rng, policy, state)
        # action, action_prob = policy()
        _, state, _, _ = step!(env, action; rng=rng)
        states[step+1] = copy(state[1])
    end
    return states
end

function string(env::ContFourRooms)
    str = fill("0", size(env)...)
    str[env.walls] .= "1"
    str[project(env, env.state)...] = "a"
    return join([join(str[i, :], " ") for i in 1:size(env)[1]], "\n")
end

Base.print(io::IO, env::ContFourRooms) = print(io, string(env))


