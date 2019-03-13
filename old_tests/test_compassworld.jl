
using GVFN

function main()

    env = CompassWorld(8,8)
    start!(env)
    println(env)
    for step in 1:100

        _, s, _, _ = step!(env, rand(get_actions(env)))
        print(env)
        println(s)
        println("")

    end


end




