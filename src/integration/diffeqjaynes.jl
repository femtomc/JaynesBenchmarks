module DiffEqJaynes

using Jaynes
using DifferentialEquations

# For adjoints.
using DiffEqSensitivity

# Generate some data with fixed parameters.
function lotka_volterra(du,u,p,t)
  x, y = u
  α, β, δ, γ = p
  du[1] = dx = (α - β * y) * x
  du[2] = dy = (δ * x - γ) * y
end

function fit_lotka_volterra()
    σ = rand(:σ, InverseGamma(2, 3))
   
    # Turing.
    # α ~ truncated(Normal(1.5,0.5),0.5,2.5)
    # β ~ truncated(Normal(1.2,0.5),0,2)
    # γ ~ truncated(Normal(3.0,0.5),1,4)
    # δ ~ truncated(Normal(1.0,0.5),0,2)
    
    α = rand(:α, Normal(1.5, 0.5))
    β = rand(:β, Normal(1.2, 0.4))
    γ = rand(:γ, Normal(3.0, 0.5))
    δ = rand(:δ, Normal(1.0, 0.5))

    p = [α, β, γ, δ]
    prob = ODEProblem(lotka_volterra, u0, (0.0, 10.0), p)
    predicted = solve(prob, saveat = 0.1)

    out = [rand(:obs => i, MvNormal(predicted[i], σ)) for i in 1 : length(predicted)]
    out
end

# Data.
p = [1.5, 1.0, 3.0, 1.0]
u0 = [1.0, 1.0]
prob = ODEProblem(lotka_volterra, u0, (0.0, 10.0), p)
data = solve(prob, Tsit5(), saveat=0.1)

# HMC.
infer = data -> begin
    sel = target([(:obs => i, ) => data[i] for i in 1 : length(data)])
    display(sel)
    sleep(0.5)
    
    @time ret, cl = generate(sel, fit_lotka_volterra)
  
    # Targets.
    addrs = [(:α, ), (:β, ), (:γ, ), (:δ, )]
    sel = target(addrs)
    calls = []
    for i in 1 : 2000
        @time cl, acc = hmc(sel, cl)
        @time cl, acc = mh(sel, cl)
        i % 30 == 0 && begin
            push!(calls, cl)
        end
    end
    for addr in addrs
        println("Est. $addr : $(mean(map(calls) do cl
                cl[addr]
            end))")
    end
end
infer(data)

end # module
