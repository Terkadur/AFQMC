using JLD
include("./src/SwapQMC.jl")
using .SwapQMC

function run_regular_sampling_gs(
    extsys::ExtendedSystem{Sys}, qmc::QMC, φ₀::Vector{Wf}, path::String, filename_pq::String, filename_sgn::String, swap_period::Int
) where {Sys<:Hubbard,Wf<:AbstractMatrix}

    system = extsys.system

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)

    replica = Replica(extsys, walker1, walker2)

    bins = qmc.measure_interval

    sampler = EtgSampler(extsys, qmc)
    avg_sgn = zeros(Float64, qmc.nsamples)

    # warm-up steps
    println("Warming up")
    for i in 1:qmc.nwarmups
        if (i - 1) % swap_period < swap_period / 2 - 1
            sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=false)
        elseif (i - 1) % swap_period == swap_period / 2 - 1
            print(i)
            print("/")
            println(qmc.nwarmups)
            sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=true)
        elseif swap_period / 2 - 1 < (i - 1) % swap_period < swap_period - 1
            sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=false)
        else
            print(i)
            print("/")
            println(qmc.nwarmups)
            sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=true)
        end
    end

    # measurements
    println("Measuring")
    for i in 1:qmc.nsamples
        # deleteat!(walker1.tmp_r, findall(x -> true, walker1.tmp_r))
        # deleteat!(walker2.tmp_r, findall(x -> true, walker2.tmp_r))
        if (i - 1) % swap_period < swap_period / 2 - 1
            sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=false)
        elseif (i - 1) % swap_period == swap_period / 2 - 1
            print(i)
            print("/")
            println(qmc.nsamples)
            sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=true)
        elseif swap_period / 2 - 1 < (i - 1) % swap_period < swap_period - 1
            sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=false)
        else
            print(i)
            print("/")
            println(qmc.nsamples)
            sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=true)
        end

        measure_Pn2!(sampler, replica, forwardMeasurement=true, forceSymmetry=qmc.forceSymmetry)

        # avg_sgn[i] = average_sign(walker1, walker2)
    end

    # store the measurement
    # jldopen("$(path)/$(filename_pq)", "w") do file
    #     write(file, "Pn2_up", sampler.Pn₊)
    #     write(file, "Pn2_dn", sampler.Pn₋)
    # end
    # jldopen("$(path)/$(filename_sgn)", "w") do file
    #     write(file, "Pn2_up", sampler.Pn₊)
    # end

    return nothing
end

function average_sign(walker1::HubbardWalker, walker2::HubbardWalker)
    if length(walker1.tmp_r) + length(walker2.tmp_r) == 0
        return 0
    end

    avg_sign = 0
    for i in walker1.tmp_r
        avg_sign += sign(i)
    end
    for i in walker2.tmp_r
        avg_sign += sign(i)
    end
    avg_sign /= length(walker1.tmp_r) + length(walker2.tmp_r)
    return avg_sign
end