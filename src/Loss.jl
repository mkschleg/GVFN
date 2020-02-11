
import Flux

tderror(v_t, c, γ_tp1, ṽ_tp1) =
    (v_t .- (c .+ γ_tp1.*ṽ_tp1))

function offpolicy_tdloss(ρ_t::Array{T, 1},
                          v_t::Flux.TrackedArray,
                          c::Array{T, 1},
                          γ_tp1::Array{T, 1},
                          ṽ_tp1::Array{T, 1}) where {T<:AbstractFloat}
    target = c + γ_tp1.*ṽ_tp1
    return sum(ρ_t.*((v_t - target).^2)) * (1//(2*length(ρ_t)))
end

function offpolicy_tdloss_gvfn(ρ_t::Array{T, 1},
                               v_t::Flux.TrackedArray,
                               c::Array{T, 1},
                               γ_tp1::Array{T, 1},
                               ṽ_tp1::Array{T, 1}) where {T<:AbstractFloat}
    return sum(ρ_t.*((v_t - c - γ_tp1.*ṽ_tp1).^2))
end

tdloss(v_t, c, γ_tp1, ṽ_tp1) =
    (1//2)*length(c)*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))


