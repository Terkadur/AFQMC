"""
    Monte Carlo Sampler
"""

### preallocated matrices for computing entanglement spectrum ###
struct EtgData{T,E}
    # entanglement Hamiltonians
    HA₁::LDR{T,E}
    HA₂::LDR{T,E}

    # temporal data for Gₐ
    GA₁::LDR{T,E}
    GA₂::LDR{T,E}

    # temporal data for I-Gₐ
    ImGA₁::LDR{T,E}
    ImGA₂::LDR{T,E}

    function EtgData(extsys::ExtendedSystem)
        LA = extsys.LA

        T = eltype(extsys.system.auxfield)
        HA₁, HA₂, GA₁, GA₂, ImGA₁, ImGA₂ = ldrs(zeros(T, LA, LA), 6)

        return new{T,Float64}(HA₁, HA₂, GA₁, GA₂, ImGA₁, ImGA₂)
    end
end

struct EtgSampler{T,E}
    # partition and measure point (space and time)
    Aidx::Vector{Int}
    mp_x::Int
    mp_t::Int
    L::Int

    # counters
    s_counter::Base.RefValue{Int} # count the number of collected samples

    # observables
    p::Vector{Float64}          # transition probability
    Pn₊::Matrix{ComplexF64}     # spin-up probability distribution
    Pn₋::Matrix{ComplexF64}     # spin-down probability distribution

    # temporal data
    data::EtgData{T,E}
    tmpPn_up::Matrix{ComplexF64}
    tmpPn_dn::Matrix{ComplexF64}

    # LDR workspace
    wsA::LDRWorkspace{T,E}
end

function EtgSampler(extsys::ExtendedSystem, qmc::QMC; nsamples=qmc.nsamples)
    Aidx = extsys.Aidx
    x = Aidx[end]
    θ = div(extsys.system.L, 2)

    p = zeros(qmc.nsamples)

    L = length(Aidx)
    Nₐ_up = min(L, extsys.system.N[1])
    Nₐ_dn = min(L, extsys.system.N[2])
    Pn₊ = zeros(ComplexF64, L + 1, nsamples)
    Pn₋ = zeros(ComplexF64, L + 1, nsamples)
    tmpPn_up = zeros(ComplexF64, L + 1, Nₐ_up)
    tmpPn_dn = zeros(ComplexF64, L + 1, Nₐ_dn)

    data = EtgData(extsys)
    wsA = ldr_workspace(data.HA₁)

    return EtgSampler(Aidx, x, θ, L, Ref(1), p, Pn₊, Pn₋, data, tmpPn_up, tmpPn_dn, wsA)
end
