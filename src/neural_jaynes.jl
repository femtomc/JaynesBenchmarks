module NeuralJaynes

using Zygote
using Flux
using Flux: Chain, Dense
using NNlib
using Jaynes

encoder = Chain(Dense(784, 32),
                Dense(32, 10), 
                softmax)

decoder = Chain(Dense(784, 32),
                Dense(32, 10), 
                softmax)

function model()
    init = rand(784)
    params = encoder(init)
    z = rand(:z, MvNormal(params))
    dec = decoder(z)
    return dec
end

ret, cl = simulate(model)
display(cl.trace)

end # module

