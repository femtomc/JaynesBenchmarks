module SossForeignModels

using Jaynes
Jaynes.@load_soss_fmi()

# A Soss model.
m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(5)
end

# Some generic function - this is where Jaynes shines!
bar = () -> begin
    x = rand(:x, Normal(5.0, 1.0))
    soss_ret = soss_fmi(:foo, m, (σ = x,))
    return soss_ret
end

ret, cl = Jaynes.simulate(bar)
display(cl.trace)
println(get_score(cl))

end # module
