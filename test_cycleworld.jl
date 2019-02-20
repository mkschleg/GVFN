

using GVFN

build_features(s) = [1.0, s[1], 1-s[1]]

function test()

    env = CycleWorld(6)

    discount = (args...)->0.0
    cumulant(i) = (s_tp1, p_tp1)-> i==1 ? s_tp1[1] : p_tp1[i-1]
    cumulants = [[cumulant(i) for i in 1:6]; [cumulant(1)]]
    discounts = [[discount for i in 1:6]; [(env_state)-> env_state[1] == 1 ? 0.0 : 0.9]]
    # println(size(discounts))

    gvflayer = GVFLayer(7, 3, cumulants, discounts)
    _, s_t = start!(env)
    h_t = zeros(7)
    h_tm1 = zeros(7)
    for step in 1:500000
        _, s_tp1, _, _ = step!(env, 1)
        println(step)
        print(env)
        h_t, h_t = gvflayer(h_tm1, build_features(s_t))
        # h_tp1, h_tp1 = gvflayer(h_t, build_features(s_tp1))
        simple_train(gvflayer, 0.8, 0.9, h_tm1, build_features(s_t), build_features(s_tp1), s_tp1)
        println(h_t)

        s_t .= s_tp1
        h_tm1 .= h_t
        # h_t .= h_tp1
    end


end




