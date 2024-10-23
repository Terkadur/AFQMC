using JLD
include("./src/SwapQMC.jl")
using .SwapQMC

function run_regular_sampling_gs(
    extsys::ExtendedSystem{Sys}, qmc::QMC, φ₀::Vector{Wf}, path::String, filename::String, swap_period::Int=256
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
        if (i - 1) % swap_period < swap_period/2 - 1
            sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=false)
        elseif (i - 1) % swap_period == swap_period/2 - 1
            print(i)
            print("/")
            println(qmc.nwarmups)
            sweep!(system, qmc, replica, walker1, 1, loop_number=1, jumpReplica=true)
        elseif swap_period/2 - 1 < (i - 1) % swap_period < swap_period - 1
            sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=false)
        else
            print(i)
            print("/")
            println(qmc.nwarmups)
            sweep!(system, qmc, replica, walker2, 2, loop_number=1, jumpReplica=true)
        end

        # print(i)
        # print("/")
        # println(qmc.nwarmups)
        # sign1 = 0
        # for i in walker1.tmp_r
        #     sign1 += sign(i)
        # end
        # sign1 /= length(walker1.tmp_r)
        # @show sign1
    end

    # measurements
    println("Measuring")
    for i in 1:qmc.nsamples
        if (i - 1) % swap_period < swap_period/2 - 1
            sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=false)
        elseif (i - 1) % swap_period == swap_period/2 - 1
            print(i)
            print("/")
            println(qmc.nsamples)
            sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=true)
        elseif swap_period/2 - 1 < (i - 1) % swap_period < swap_period - 1
            sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=false)
        else
            print(i)
            print("/")
            println(qmc.nsamples)
            sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=true)
        end

        measure_Pn2!(sampler, replica, forwardMeasurement=true, forceSymmetry=qmc.forceSymmetry)

        # sign1 = 0
        # for i in walker1.tmp_r
        #     sign1 += sign(i)
        # end
        # sign1 /= length(walker1.tmp_r)
        # @show sign1
    end

    # store the measurement
    jldopen("$(path)/$(filename)", "w") do file
        write(file, "Pn2_up", sampler.Pn₊)
        write(file, "Pn2_dn", sampler.Pn₋)
    end

    # sign2 = 0
    # for i in walker2.tmp_r
    #     sign2 += sign(i)
    # end
    # sign2 /= length(walker2.tmp_r)
    # @show sign2


    return nothing
end
