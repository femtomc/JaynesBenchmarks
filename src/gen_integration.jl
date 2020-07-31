module GenForeignModels

using Jaynes
Jaynes.@load_gen_fmi()
using Gen

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

end # module
