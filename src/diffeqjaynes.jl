module DiffEqJaynes

using Jaynes
using DifferentialEquations

# For adjoints.
using DiffEqSensitivity

# Generate some data with fixed parameters.
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β*y)x
  du[2] = dy = (δ*x - γ)y
end
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0,1.0]
prob = ODEProblem(lotka_volterra,u0,(0.0,10.0),p)
data = Array(solve(prob,Tsit5(),saveat=0.1))

function fit_lotka_volterra()
    σ = rand(:σ, InverseGamma(2, 3))
    α = rand(:α, truncated(Normal(1.5, 0.5), 0.5, 2.5))
    β  = rand(:β, truncated(Normal(1.2, 0.5), 0, 2))
    γ = rand(:γ, truncated(Normal(3.0, 0.5), 1, 4))
    δ = rand(:δ, truncated(Normal(1.0, 0.5), 0, 2))

    p = [α, β, γ, δ]
    prob = ODEProblem(lotka_volterra, u0, (0.0, 10.0), p)
    predicted = solve(prob, saveat=0.1; sensealg = ZygoteAdjoint())

    for i = 1:length(predicted)
        rand(:obs => i, MvNormal(predicted[i], σ))
    end
end

ret, cl = simulate(fit_lotka_volterra)
display(cl.trace)

# Inference.
addrs = [(:α, ), (:β, ), (:γ, ), (:σ, ), (:δ, )]
sel = selection(addrs)

new, acc = hmc(sel, cl)

end # module
