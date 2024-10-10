using JLD
include("./src/SwapQMC.jl")
using .SwapQMC

function run_regular_sampling_gs(
    extsys::ExtendedSystem{Sys}, qmc::QMC, φ₀::Vector{Wf}, path::String, filename::String
) where {Sys<:Hubbard,Wf<:AbstractMatrix}

    system = extsys.system

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)

    replica = Replica(extsys, walker1, walker2)

    bins = qmc.measure_interval

    sampler = EtgSampler(extsys, qmc)

    # warm-up steps
    println("Warming up")
    for i in 1:qmc.nwarmups
        if (i - 1) % 16 < 7
            sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=false)
        elseif (i - 1) % 16 == 7
            sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=true)
        elseif 7 < (i - 1) % 16 < 15
            sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=false)
        else
            sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=true)
        end

        print(i)
        print("/")
        println(qmc.nwarmups)
    end

    # measurements
    println("Measuring")
    for i in 1:qmc.nsamples
        if (i - 1) % 16 < 7
            sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=false)
        elseif (i - 1) % 16 == 7
            sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=true)
        elseif 7 < (i - 1) % 16 < 15
            sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=false)
        else
            sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=true)
        end

        measure_Pn2!(sampler, replica, forwardMeasurement=true)

        print(i)
        print("/")
        println(qmc.nsamples)
    end

    # store the measurement
    jldopen("$(path)/$(filename)", "w") do file
        write(file, "Pn2_up", sampler.Pn₊)
        write(file, "Pn2_dn", sampler.Pn₋)
    end

    return nothing
end
