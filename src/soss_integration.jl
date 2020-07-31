module SossForeignModels

using Jaynes
Jaynes.@load_soss_fmi()

m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(1000)
end

bar = () -> begin
    x = rand(:x, Normal(5.0, 1.0))
    return soss_fmi(:foo, m, (σ = x,))
end

ret, cl = Jaynes.simulate(bar)
display(cl.trace)

end # module
