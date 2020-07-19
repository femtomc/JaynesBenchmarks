module HMMPF_Turing

using Turing
using Jaynes
using BenchmarkTools
using Plots
using Random

Random.seed!(314159)
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
    init_obs = selection((:z0, 0.0))
    ps = initialize_filter(init_obs, 1000, kernelize, (1, ))
    for i in 2 : length(obs)
        sel = selection((:k => i => :x, obs[i - 1]))

        # Complexity of filter step is constant as a size of the trace.
        filter_step!(sel, ps, Jaynes.NoChange(), (i,))
    end
    return ps
end

# ------------ Turing ------------ #

@model TuringHMM(x::Vector{Float64}) = begin
    N = length(x)
    z = tzeros(Float64, N - 1)
    z_0 ~ Normal(0.0, 1.0)
    z[1] = z_0
    for i in 2 : N
        z[i] ~ Normal(z[i - 1], 1.0)
        x[i] ~ Normal(z[i], 10.0)
    end
end

t_simulation = obs -> begin
    init_obs = selection((:z0, 0.0))
    ps = initialize_filter(init_obs, 1000, kernelize, (1, ))
    for i in 2 : length(obs)
        sel = selection((:k => i => :x, obs[i - 1]))

        # Complexity of filter step is constant as a size of the trace.
        filter_step!(sel, ps, Jaynes.NoChange(), (i,))
    end
    return ps
end

function benchmark(t::Int)
    # Some set of observations.
    obs = [rand() for i in 1 : t]

    simulation(obs)
    t_start = time_ns()
    ps = simulation(obs)
    t_time = (time_ns() - t_start) / 1e9
    t_lmle = lot_ml_estimate(ps)

    j_simulation(obs)
    t_start = time_ns()
    ps = j_simulation(obs)
    j_time = (time_ns() - t_start) / 1e9
    j_lmle = ps.lmle
    t_time, j_time, t_lmle, j_lmle
end

steps = [1, 3, 10, 30, 50, 100, 150, 200, 500, 800]
times = map(steps) do t
    benchmark(t)
end

println(times)
l = @layout [a ; c]
t_plt = plot(steps, [t[1] for t in times], title = "Time vs. unroll steps", label = "(Gen) Unfold combinator + static DSL", legend = :topleft)
plot!(t_plt, steps, [t[2] for t in times], label = "(Jaynes) Markov-specialized call site")
plt_lmle = plot(steps, [t[3] for t in times], title = "Log marginal likelihood of data", label = "Gen")
plot!(plt_lmle, steps, [t[4] for t in times], label = "Jaynes")
savefig(plot(t_plt, plt_lmle, layout = l, palette = cgrad.([:grays :blues :heat :lightrainbow]), bt_inside = [:orange :pink :darkblue :black]), "benchmark_images/benchmark_hmmpf_$(String(gensym("gen"))).svg")


end # module
