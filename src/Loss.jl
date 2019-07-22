
tderror(v_t, c, γ_tp1, ṽ_tp1) =
    (v_t .- (c .+ γ_tp1.*ṽ_tp1))

function offpolicy_tdloss(ρ_t::Array{T, 1}, v_t::TrackedArray, c::Array{T, 1}, γ_tp1::Array{T, 1}, ṽ_tp1::Array{T, 1}) where {T<:AbstractFloat}
    target = T.(c .+ γ_tp1.*ṽ_tp1)
    return (T(0.5))*sum(ρ_t.*((v_t .- target).^2)) * (1//length(ρ_t))
end

tdloss(v_t, c, γ_tp1, ṽ_tp1) =
    0.5*Flux.mse(v_t, Flux.data(c .+ γ_tp1.*ṽ_tp1))


