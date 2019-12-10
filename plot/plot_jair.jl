


include("plot/plot_final.jl")



ring_world_sens_all() = 
    plot_sens_files(
        ["final_ringworld_gvfn_action_rtd/collected_sens_all/collect_horde_gamma_chain.jld2",
         "final_ringworld_gvfn_action_rtd/collected_sens_all/collect_horde_half_chain.jld2",
         "final_ringworld_rnn_action/collected_sens_all/collect_.jld2",
         # "final_ringworld_rnn/collected_sens_all/collect_cell_GRU.jld2"
         ],
        [[:color=>:blue, :label=>:gamma_chain],
         [:color=>:blue, :label=>:half_chain, :ls=>:dash],
         [:color=>:green, :label=>:rnn_action],
         [:color=>:green, :ls=>:dash, :label=>:GRU]],
        "ringworld_sens.pdf";
        ylims=(0,0.3), lw=2, grid=false, tickfontsize=16, tickdir=:out)

ring_world_sens_end() = 
    plot_sens_files(
        ["final_ringworld_gvfn_action_rtd/collected_sens/collect_horde_gamma_chain.jld2",
         "final_ringworld_gvfn_action_rtd/collected_sens/collect_horde_half_chain.jld2",
         "final_ringworld_rnn_action/collected_sens/collect_.jld2",
         # "final_ringworld_rnn/collected_sens_/collect_cell_GRU.jld2"
         ],
        [[:color=>:blue, :label=>:gamma_chain],
         [:color=>:blue, :label=>:half_chain, :ls=>:dash],
         [:color=>:green, :label=>:rnn_action],
         [:color=>:green, :ls=>:dash, :label=>:GRU]],
        "ringworld_sens_end.pdf";
        ylims=(0,0.3), lw=2, grid=false, tickfontsize=16, tickdir=:out)

ring_world_lc_all() =
    plot_lc_files(
        ["final_ringworld_gvfn_action_rtd/collected/truncation_2_horde_gamma_chain_optparams_[0.1].jld2",
         "final_ringworld_gvfn_action_rtd/collected/truncation_4_horde_half_chain_optparams_[0.15].jld2",
         "final_ringworld_rnn_action/collected/truncation_2_optparams_[0.15].jld2",
         # "final_ringworld_rnn/collected/truncation_32_cell_GRU_optparams_[0.0296296].jld2"
         ],
        [[:color=>:blue, :label=>"gvfn"],
         [:color=>:blue, :label=>:half_chain, :ls=>:dash],
         [:color=>:green, :label=>"rnn"],
         [:color=>:green, :ls=>:dash, :label=>:GRU]];
        save_file="ringworld_lc.pdf",
        n=1000,
        clean_func=v->sqrt.(mean(v.^2; dims=2))[1:300000],
        lw=2, grid=false, tickdir=:out, tickfontsize=16)
