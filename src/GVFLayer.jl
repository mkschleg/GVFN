


using Flux


sigmoid(x::Float64) = 1.0/(1.0 + exp(-x))
sigmoidprime(x::Float64) = sigmoid(x)*(1.0-sigmoid(x))

# Assuming single layer for now...
mutable struct GVFLayer
    weights::Array{Float64, 2} #
    traces::Array{Float64, 2}
    cumulants::Array{Any, 1}
    discounts::Array{Any, 1}

    GVFLayer(num_gvfs, num_ext_features, cumulants, discounts) =
        new(0.01.*rand(num_gvfs, num_ext_features+num_gvfs), zeros(num_gvfs, num_ext_features+num_gvfs), cumulants, discounts)
end

function (m::GVFLayer)(h, x)
    # println(size([h;x]))
    # println(size(m.weights))
    new_h = sigmoid.(m.weights*[h;x])
    return new_h, new_h
end

function simple_train(gvfn::GVFLayer, α, λ, h_tm1, x_t, x_tp1, env_state_tp1)

    

    preds_t, preds_t = gvfn(h_tm1, x_t)
    # println(preds_t)
    preds_tp1, preds_tp1 = gvfn(preds_t, x_tp1)
    # println(gvfn.discounts)
    cumulants = [gvfn.cumulants[i](env_state_tp1, preds_tp1) for i in 1:length(gvfn.cumulants)]
    discounts = [gvfn.discounts[i](env_state_tp1) for i in 1:length(gvfn.cumulants)]

    println(cumulants)
    println(discounts)

    targets = cumulants + discounts.*preds_tp1
    δ = targets - preds_t

    @inbounds for gvf in 1:length(discounts)
        trace_view = view(gvfn.traces, gvf, :)
        trace_view .= 1.0.*(discounts[gvf].*λ.*trace_view .+ sigmoidprime(preds_t[gvf]).*[h_tm1;x_t])
        gvfn.weights[gvf,:] .+= trace_view.*(α*δ[gvf])
    end
    # gvfn.weights .+= gvfn.traces.*(α.*δ)
end


