


# function tdlambda!(weights, traces, v, v_prime, ϕ_t, ϕ_tp1, r, γ_t, γ_tp1, λ, α, ρ)
#     δ = r + γ_tp1*v_prime - v
#     traces .= 
# end




function optimize_gvfs!(preds::Array{Float64, 1},
                        preds_tilde::Array{Float64, 1},
                        weights::Array{Array{Float64, 1}, 1},
                        traces::Array{Array{Float64, 1}, 1},
                        ϕ_t::Array{Float64, 1},
                        ϕ_tp1::Array{Float64, 1},
                        r::Array{Float64, 1},
                        γ_t::Array{Float64, 1},
                        γ_tp1::Array{Float64, 1},
                        λ::Float64,
                        α::Float64,
                        numgvfs::Integer)
    # Threads.@threads for gvf in 1:numgvfs
    for gvf in 1:numgvfs
        δ = r[gvf] + γ_tp1[gvf]*preds_tilde[gvf] - preds[gvf]
        traces[gvf] .= 1.0.*((γ_t[gvf]*λ).*traces[gvf] .+ sigmoidprime(dot(weights[gvf], ϕ_t)).*ϕ_t)
        weights[gvf] .+= (α*δ).*traces[gvf]
    end
end



