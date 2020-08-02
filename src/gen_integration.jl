module GenForeignModels

using Jaynes
Jaynes.@load_gen_fmi()

@gen (static) function foo(z::Float64)
    x = @trace(normal(z, 1.0), :x)
    y = @trace(normal(x, 1.0), :y)
    return x
end

Gen.load_generated_functions()

bar = () -> begin
    x = rand(:x, Normal(0.0, 1.0))
    return gen_fmi(:foo, foo, x)
end

ret, cl = Jaynes.simulate(bar)
display(cl.trace)

sel = selection([(:foo, :x) => 5.0])
ret, cl, w = Jaynes.update(sel, cl)
display(cl.trace)

sel = selection([(:foo, :x)])
ret, cl, w, rd, d = Jaynes.regenerate(sel, cl)
display(cl.trace)

sel = selection([(:x, ) => 5.0,
                 (:foo, :x) => 10.0,
                 (:foo, :y) => 10.0])
ret, score = Jaynes.score(sel, bar)
println(score)

end # module
