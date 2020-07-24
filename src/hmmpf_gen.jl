module HMMPF_Gen

using Gen
using Jaynes
using BenchmarkTools
using Plots
gr()

# ------------ Jaynes ------------ #

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
    init_obs = selection([(:z0, ) => 0.0])
    ps = initialize_filter(init_obs, 1000, kernelize, (1, ))
    for i in 2 : length(obs)
        sel = selection([(:k, i, :x) => obs[i - 1]])

        # Complexity of filter step is constant as a size of the trace.
        filter_step!(sel, ps, Jaynes.NoChange(), (i,))
        check_ess_resample!(ps)
    end
    return ps
end

# ------------ Gen ------------ #

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
    return (init_state, states)
end

Gen.load_generated_functions()

simulation = obs -> begin
    init_obs = Gen.choicemap((:z0, 0.0))
    ps = initialize_particle_filter(g_sim, (0, ), init_obs, 1000)
    for t in 1 : length(obs) - 1
        sel = Gen.choicemap((:chain => t => :x, obs[t]))
        #sel = Gen.choicemap()

        # Complexity of filter step is constant as a size of the trace.
        Gen.particle_filter_step!(ps, (t,), (UnknownChange(),), sel)
        Gen.maybe_resample!(ps)
    end
    return ps
end


function benchmark(t::Int)
    # Some set of observations.
    obs = [rand() for i in 1 : t]

    t_start = time_ns()
    ps = simulation(obs)
    g_time = (time_ns() - t_start) / 1e9
    g_lmle = log_ml_estimate(ps)

    t_start = time_ns()
    ps = j_simulation(obs)
    j_time = (time_ns() - t_start) / 1e9
    j_lmle = get_lmle(ps)
    g_time, j_time, g_lmle, j_lmle
end

#steps = collect(1 : 5 : 500) .+ 4
#prepend!(steps, 1)
steps = [1, 5, 10, 20, 50, 100, 200, 500, 800, 1000, 1500]

# Warm up.
obs = [rand() for i in 1 : 10]
simulation(obs)
j_simulation(obs)

times = map(steps) do t
    println("T -> $t")
    @time benchmark(t)
end

println(times)
l = @layout [a ; c]
g_plt = plot(steps, [t[1] for t in times], title = "Time vs. unroll steps", label = "(Gen) Unfold combinator + static DSL", legend = :topleft)
plot!(g_plt, steps, [t[2] for t in times], label = "(Jaynes) Markov-specialized call site")
plt_lmle = plot(steps, [t[3] for t in times], title = "Log marginal likelihood of data", label = "Gen")
plot!(plt_lmle, steps, [t[4] for t in times], label = "Jaynes")
savefig(plot(g_plt, plt_lmle, layout = l, palette = cgrad.([:grays :blues :heat :lightrainbow]), bg_inside = [:orange :pink :darkblue :black]), "benchmark_images/benchmark_hmmpf_$(String(gensym("gen"))).svg")

end # module
