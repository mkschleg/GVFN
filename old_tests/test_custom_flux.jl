
using Flux
using Flux.Tracker
using Flux: @epochs


mutable struct CustomLayer
    weights::TrackedArray
    b::TrackedArray
    CustomLayer(in::Integer, out::Integer) = new(param(zeros(out, in)), param(zeros(out)))
end

# Currently Linear
(m::CustomLayer)(x) = m.weights.data*x .+ m.b.data
tracked_predict(m::CustomLayer, x) = m.weights*x .+ m.b
Flux.@treelike CustomLayer

function test_custom()

    cl_model = Chain(
        Dense(5,5),
        CustomLayer(10,5)
    )

    model = Chain(
        cl_model,
        Dense(5,5),
        Dense(5,5)
    )

    # model = CustomLayer(10, 5)


    function loss(x, y)
        Flux.mse(model(x), y)
    end

    function cl_loss(x, y)
        Flux.mse(tracked_predict(model[1], x), y)
    end

    CustomLayer_opt = rand(10,5)
    CustomLayer_opt_b = rand(5)

    W_opt = rand(10,5)
    b_opt = rand(5)
    

    function get_opt(x, W, b)
        W'*x .+ b
    end

    X = [rand(10) for i = 1:1000]
    Y = [get_opt(X[i], W_opt, b_opt) for i = 1:1000]
    Y_cl = [get_opt(X[i], CustomLayer_opt, CustomLayer_opt_b) for i = 1:1000]
    data = zip(X, Y)
    data_cl = zip(X, Y_cl)

    test_X = rand(10, 100)
    test_Y = get_opt(test_X, W_opt, b_opt)
    test_Y_cl = get_opt(test_X, CustomLayer_opt, CustomLayer_opt_b)

    function evalcb()
        @show(loss(test_X, test_Y))
        @show(cl_loss(test_X, test_Y_cl))
    end

    opt = ADAM(0.0001)
    opt_cl = Descent(0.001)
    function epoch()
        Flux.train!(loss, params(model), data, opt)
        Flux.train!(cl_loss, params(model), data_cl, opt)
        evalcb()
    end

    @epochs 100 epoch()

    for epoch in 1:100
        println("Epoch: $(epoch)")
        Flux.train!(loss, params(model), data, opt)
        Flux.train!(cl_loss, params(model), data_cl, opt)

        evalcb()
    end

    println(CustomLayer_opt)
    println(model[1].weights)


end

function test_flux()

    model = CustomLayer(5,2)

    W = param(rand(2, 5))
    b = param(rand(2))

    predict(x) = W*x .+ b
    loss(x, y) = Flux.mse(tracked_predict(model, x), y)

    W_opt = rand(2,5)
    b_opt = rand(2)

    get_opt(x) = W_opt*x .+ b_opt .+ 0.001*randn(length(size(x)) == 1 ? 2 : (2, size(x)[2]))

    X = [rand(5) for i = 1:1000]
    Y = [get_opt(X[i]) for i = 1:1000]
    data = zip(X, Y)

    train_X = rand(5, 100)
    train_Y = get_opt(train_X)

    evalcb() = @show(loss(train_X, train_Y))

    opt = ADAM(0.001)
    @epochs 50 Flux.train!(loss, params(model), data, opt, cb = Flux.throttle(evalcb, 5))

end
