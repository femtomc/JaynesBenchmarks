module GenForeignModels

using Jaynes
Jaynes.@load_flux_fmi()

bar = model -> begin
    q = learnable(:q)
    z = foreign(:z, model, q)
    y = rand(:y, Normal(z, 1.0))
    return y
end

model = Chain(Dense(10, 5), Dense(5, 5)); 

ps = parameters([(:q, ) => ones(10)])
ret, cl = Jaynes.simulate(ps, bar, model)

end # module
