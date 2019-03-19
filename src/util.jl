

using Flux
using Flux.Tracker

glorot_uniform(rng::Random.AbstractRNG, dims...) = (rand(rng, Float32, dims...) .- 0.5f0) .* sqrt(24.0f0/sum(dims))
glorot_normal(rng::Random.AbstractRNG, dims...) = randn(rng, Float32, dims...) .* sqrt(2.0f0/sum(dims))

function jacobian(δ, pms)
    k  = length(δ)
    J = IdDict()
    for id in pms
        v = get!(J, id, zeros(k, size(id)...))
        for i = 1:k
            Flux.back!(δ[i], once = false) # Populate gradient accumulator
            v[i, :,:] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
    J
end

function jacobian!(J::IdDict, δ::TrackedArray, pms::Params)
    k  = length(δ)
    for i = 1:k
        Flux.back!(δ[i], once = false) # Populate gradient accumulator
        for id in pms
            v = get!(J, id, zeros(typeof(id[1].data), k, size(id)...))::Array{typeof(id[1].data), 3}
            v[i, :, :] .= id.grad
            id.grad .= 0 # Reset gradient accumulator
        end
    end
end

# export ArgIterator, iterate, next_state!
struct ArgIterator
    dict::Dict
    stable_arg::Vector{String}
    arg_list::Vector{String}
    done::Bool
    ArgIterator(dict, stable_arg) = new(dict, stable_arg, collect(keys(dict)), false)
end

function Base.iterate(iter::ArgIterator)
    state = ones(Int64, length(iter.arg_list))
    new_ret_list = Vector{String}()

    for (arg_idx, arg) in enumerate(iter.arg_list)
        push!(new_ret_list, arg)
        push!(new_ret_list, string(iter.dict[arg][state[arg_idx]]))
    end

    return [new_ret_list; iter.stable_arg], next_state!(iter, state)
end

function next_state!(iter::ArgIterator, state)

    state[end] += 1

    for (arg_idx, arg) in Iterators.reverse(enumerate(iter.arg_list))
        if arg_idx == 1
            return state
        end

        if state[arg_idx] > length(iter.dict[arg])
            state[arg_idx] = 1
            state[arg_idx - 1] += 1
        end

    end
end


function Base.iterate(iter::ArgIterator, state)
    if state[1] > length(iter.dict[iter.arg_list[1]])
        return nothing
    end
    ret = Vector{String}()
    for (arg_idx, arg) in enumerate(iter.arg_list)
        push!(ret, arg)
        push!(ret, string(iter.dict[arg][state[arg_idx]]))
    end

    new_state = next_state!(iter, state)
    return [ret; iter.stable_arg], new_state
end


