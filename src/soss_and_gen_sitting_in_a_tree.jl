module SossForeignModels

using Jaynes
Jaynes.@load_soss_fmi()
Jaynes.@load_gen_fmi()

# A Soss model.
m = @model σ begin
    μ ~ Normal()
    y ~ Normal(μ, σ) |> iid(5)
end

@gen function foo(x::Float64)
    y = @trace(normal(x, 1.0), :y)
    return y
end

# Some generic function - this is where Jaynes shines!
bar = () -> begin
    x = rand(:x, Normal(5.0, 1.0))
    gen_ret = gen_fmi(:gen, foo, x)
    soss_ret = soss_fmi(:foo, m, (σ = x,))
    return soss_ret
end

ret, cl = Jaynes.simulate(bar)
display(cl.trace)
println(get_score(cl))

end # module
