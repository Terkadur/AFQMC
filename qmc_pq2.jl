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
    sgn = zeros(Float64, qmc.nsamples)

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
    if qmc.saveRatio
        for i in 1:qmc.nsamples
            if (i - 1) % swap_period < swap_period / 2 - 1
                deleteat!(walker1.tmp_r, findall(x -> true, walker1.tmp_r))
                sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=false)
                sgn[i] = average_sign(walker1)
            elseif (i - 1) % swap_period == swap_period / 2 - 1
                print(i)
                print("/")
                println(qmc.nsamples)
                deleteat!(walker1.tmp_r, findall(x -> true, walker1.tmp_r))
                sweep!(system, qmc, replica, walker1, 1, loop_number=bins, jumpReplica=true)
                sgn[i] = average_sign(walker1)
            elseif swap_period / 2 - 1 < (i - 1) % swap_period < swap_period - 1
                deleteat!(walker2.tmp_r, findall(x -> true, walker2.tmp_r))
                sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=false)
                sgn[i] = average_sign(walker2)
            else
                print(i)
                print("/")
                println(qmc.nsamples)
                deleteat!(walker2.tmp_r, findall(x -> true, walker2.tmp_r))
                sweep!(system, qmc, replica, walker2, 2, loop_number=bins, jumpReplica=true)
                sgn[i] = average_sign(walker2)
            end

            try
                measure_Pn2!(sampler, replica, forwardMeasurement=true, forceSymmetry=qmc.forceSymmetry)
            catch
                print("Pn2 measurement failed on sample ")
                println(i)
                println(sampler.s_counter)
            end
        end
    else
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

            try
                measure_Pn2!(sampler, replica, forwardMeasurement=true, forceSymmetry=qmc.forceSymmetry)
            catch
                print("Pn2 measurement failed on sample ")
                println(i)
            end
        end
    end

    # store the measurement
    jldopen("$(path)/$(filename_pq)", "w") do file
        write(file, "Pn2_up", sampler.Pn₊)
        write(file, "Pn2_dn", sampler.Pn₋)
    end
    if qmc.saveRatio
        jldopen("$(path)/$(filename_sgn)", "w") do file
            write(file, "Sgn", sgn)
        end
    end

    return nothing
end

function average_sign(walker::HubbardWalker)
    avg_sign = 0
    for i in walker.tmp_r
        avg_sign += sign(i)
    end
    avg_sign /= length(walker.tmp_r)
    return avg_sign
end