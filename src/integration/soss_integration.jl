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
    soss_ret = foreign(:foo, m, (σ = x,))
    return soss_ret
end

ret, cl = Jaynes.simulate(m, (σ = 5.0,))

sel = (μ = 3.0, σ = 5.0)
ret, cl, w = Jaynes.generate(sel, m, (σ = 5.0, ))
display(cl.trace)
println(get_score(cl))

sel = selection([(:μ, ), (:σ, )])
ret, cl, w, rd, d = Jaynes.regenerate(sel, cl)
display(cl.trace)
println(get_score(cl) - w)

end # module
