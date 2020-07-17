module JaynesBenchmarks

using Gen
using Jaynes
using BenchmarkTools

# Markov kernel.
function kernel(prev_latent::Float64)
    z = rand(:z, Normal(prev_latent, 1.0))
    x = rand(:x, Normal(z, 10.0))
    return z
end

# Specialized Markov call site informs tracer of dependency information.
kernelize = n -> begin
    initial_latent = rand(:z0, Normal(0.0, 1.0))
    markov(:k, kernel, n, initial_latent)
end

# Note - off-by-one difference for markov specialize vs. Gen - try and fix later, I think it's irrelevant from a performance perspective.
j_simulation = obs -> begin
    ps = initialize_filter(selection(), 1000, kernelize, (1, ))
    for i in 2:500
        sel = selection((:k => i => :x, obs[i - 1]))

        # Complexity of filter step is constant as a size of the trace.
        filter_step!(sel, ps, Jaynes.NoChange(), (i,))
    end
    return ps
end

# State for Unfold.
struct State
    z::Float64
    x::Float64
end

@gen (static) function g_kernel(t::Int, prev_state::State)
    z = @trace(normal(prev_state.z, 1.0), :z)
    x = @trace(normal(z, 10.0), :x)
    next_state = State(z, x)
    return next_state
end

chain = Gen.Unfold(g_kernel)

# Full simulation.
@gen (static) function g_sim(t::Int)
    z_0 = @trace(normal(0.0, 1.0), :z0)
    init_state = State(z_0, 0.0)
    states = @trace(chain(t, init_state), :chain)
    return [init_state, states...]
end

Gen.load_generated_functions()

simulation = obs -> begin
    ps = initialize_particle_filter(g_sim, (0, ), choicemap(), 1000)
    for i in 1:499
        sel = Gen.choicemap((:chain => i => :x, obs[i]))

        # Complexity of filter step is constant as a size of the trace.
        particle_filter_step!(ps, (i, ), (), sel)
    end
    return ps
end

# Some set of observations.
obs = [rand() for i in 1:499]

println("Gen:")
simulation(obs)
ps = @btime simulation(obs)
println(log_ml_estimate(ps))
GC.gc()
println("Jaynes:")
j_simulation(obs)
ps = @btime j_simulation(obs)
println(ps.lmle)

end # module
