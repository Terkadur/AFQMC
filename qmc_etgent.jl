using JLD
include("./src/SwapQMC.jl")
using .SwapQMC

function run_incremental_sampling_gs(
    extsys::ExtendedSystem{Sys}, qmc::QMC, φ₀::Vector{Wf}, λₖ::Float64, Nₖ::Int,
    path::String, filename::String, swap_period::Int
) where {Sys<:Hubbard,Wf<:AbstractMatrix}

    system = extsys.system

    walker1 = HubbardWalker(system, qmc, φ₀)
    walker2 = HubbardWalker(system, qmc, φ₀)

    replica = Replica(extsys, walker1, walker2, λₖ=λₖ)

    bins = qmc.measure_interval

    detgA = zeros(qmc.nsamples)
    sgndetgA = zeros(qmc.nsamples)
    sgnprob = zeros(qmc.nsamples)

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

        if qmc.forceSymmetry
            detgA[i] = exp(-2 * replica.logdetGA_up[] / Nₖ)
        else
            detgA[i] = real(exp(-(replica.logdetGA_up[] + replica.logdetGA_dn[]) / Nₖ))
            sgndetgA[i] = real(replica.sgnlogdetGA_up[] * replica.sgnlogdetGA_dn[])
        end

        sgnprob[i] = real(replica.sgnprob[])
    end

    if mean(sgnprob) < 0
        sgnprob .*= -1
    end

    # store the measurement
    jldopen("$(path)/$(filename)", "w") do file
        write(file, "sgnprob", sgnprob)
        write(file, "detgA", detgA)
        write(file, "sgndetgA", sgndetgA)
    end

    return nothing
end
