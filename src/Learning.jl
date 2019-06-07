


struct RTD
end

# update!(model, horde::AbstractHorde, opt, lu::TD, state_seq, env_state_tp1, action_t=nothing, b_prob=1.0; prms=nothing)

function update!(gvfn::Flux.Recur{T}, horde::AbstractHorde,
                 out_model, out_horde::AbstractHorde,
                 opt, lu::RTD, h_init,
                 state_seq, env_state_tp1, action_t=nothing, b_prob=1.0) where {T}

    update!(gvfn, opt, lu, h_init, state_seq, env_s_tp1)

    reset!(gvfn, h_init)
    preds = gvfn.(state_seq)

    update!(out_model, out_horde, opt, lu, Flux.data.(preds), env_s_tp1)

    out_preds = out_model(preds[end])

    return preds, out_preds

end


# function update!(rnn::Flux.Recur{T}, horde::AbstractHorde,
#                  out_model, out_horde::AbstractHorde,
#                  opt, lu::RTD,
#                  state_seq, env_state_tp1, action_t=nothing, b_prob=1.0) where {T}

#     update!(agent.gvfn, agent.opt, agent.lu, agent.hidden_state_init, agent.state_list, env_s_tp1)

#     reset!(agent.gvfn, agent.hidden_state_init)
#     preds = agent.rnn.(agent.state_list)

#     update!(agent.model, agent.out_horde, agent.opt, agent.lu, Flux.data.(preds), env_s_tp1)

#     out_preds = agent.model(preds[end])

#     return preds, rnn_out

# end

