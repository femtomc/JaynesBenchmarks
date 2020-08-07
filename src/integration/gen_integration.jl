module GenForeignModels

using Jaynes
Jaynes.@load_gen_fmi()

@gen function model()
    if ({:z} ~ bernoulli(0.5))
        m1 = ({:m1} ~ gamma(1, 1))
        m2 = ({:m2} ~ gamma(1, 1))
    else
        m = ({:m} ~ gamma(1, 1))
        (m1, m2) = (m, m)
    end
    {:y1} ~ normal(m1, 0.1)
    {:y2} ~ normal(m2, 0.1)
end

function merge_means(m1, m2)
    m = sqrt(m1 * m2)
    dof = m1 / (m1 + m2)
    (m, dof)
end

function split_mean(m, dof)
    m1 = m * sqrt((dof / (1 - dof)))
    m2 = m * sqrt(((1 - dof) / dof))
    (m1, m2)
end

@gen function fixed_structure_proposal(trace)
    if trace[:z]
        {:m1} ~ normal(trace[:m1], 0.1)
        {:m2} ~ normal(trace[:m2], 0.1)
    else
        {:m} ~ normal(trace[:m], 0.1)
    end
end

@gen function split_merge_proposal(trace)
    if trace[:z]
        # currently two segments, switch to one
    else
        # currently one segment, switch to two
        {:dof} ~ uniform_continuous(0, 1)
    end
end

@involution function split_merge_involution(model_args, proposal_args, proposal_retval)

    if @read_discrete_from_model(:z)

        # currently two segments, switch to one
        @write_discrete_to_model(:z, false)
        m1 = @read_continuous_from_model(:m1)
        m2 = @read_continuous_from_model(:m2)
        (m, dof) = merge_means(m1, m2)
        @write_continuous_to_model(:m, m)
        @write_continuous_to_proposal(:dof, dof)

    else

        # currently one segments, switch to two
        @write_discrete_to_model(:z, true)
        m = @read_continuous_from_model(:m)
        dof = @read_continuous_from_proposal(:dof)
        (m1, m2) = split_mean(m, dof)
        @write_continuous_to_model(:m1, m1)
        @write_continuous_to_model(:m2, m2)
    end
end

fixed_structure_kernel(trace) = Gen.mh(trace, fixed_structure_proposal, ())
split_merge_kernel(trace) = Gen.mh(trace, split_merge_proposal, (), split_merge_involution)

combination_kernel(trace) = begin
    trace, acc1 = split_merge_kernel(trace)
    trace, acc2 = fixed_structure_kernel(trace)
    trace, acc1 || acc2
end

sel = selection([(:y1, ) => 1.0,
                 (:y2, ) => 1.3,
                 (:z, ) => false,
                 (:m, ) => 1.2])
test = () -> begin
    ret, cl = Jaynes.generate(sel, model)
    display(cl.trace)
    z_array = Vector{Bool}(undef, 1000)
    for i in 1 : 1000
        @time cl, _ = ex((), cl, combination_kernel)
        z_array[i] = get_ret(cl[:z])
    end
    z_array
end

zs = test()
println(zs)

end # module
