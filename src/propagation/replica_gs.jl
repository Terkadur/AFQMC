"""
    Replica Monte Carlo sweep in the Z_{A, 2} space, ground state
"""

"""
    sweep!(system, qmc, replica)

    Sweep a replica (two copies of walker) through the imaginary time, with the workflow
    1) update walker1 from θ to 2θ
    1) update walker2 from θ to 2θ
    1) update walker1 from θ to 0
    1) update walker2 from θ to 0
"""
function sweep!(system::Hubbard, qmc::QMC, replica::Replica)
    Θ = div(qmc.K, 2)

    walker1 = replica.walker1
    walker2 = replica.walker2

    if system.useChargeHST

        sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ+1:2Θ))
        jump_replica!(replica, 1)

        sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ+1:2Θ))
        jump_replica!(replica, 2)

        sweep!_symmetric(system, qmc, replica, walker1, 1, collect(Θ:-1:1))
        jump_replica!(replica, 1)

        sweep!_symmetric(system, qmc, replica, walker2, 2, collect(Θ:-1:1))
        jump_replica!(replica, 2)

        return nothing
    end
end

"""
    sweep!(system, qmc, replica, walker, ridx, loop_number=1, jumpReplica=false)

    Sweep a certain walker of two copies (indexed by ridx) through the imaginary time in loop_number (by default 1) 
    times from θ to 2θ and from θ to 0
"""
function sweep!(
    system::Hubbard, qmc::QMC,
    replica::Replica, walker::HubbardWalker, ridx::Int;
    loop_number::Int=1, jumpReplica::Bool=false
)
    Θ = div(qmc.K, 2)

    if system.useChargeHST || qmc.forceSymmetry
        for i in 1:loop_number
            sweep!_symmetric(system, qmc, replica, walker, ridx, collect(Θ+1:2Θ))
            sweep!_symmetric(system, qmc, replica, walker, ridx, collect(Θ:-1:1))
        end

        jumpReplica && jump_replica!(replica, ridx)
        return nothing
    else
        for i in 1:loop_number
            sweep!_asymmetric(system, qmc, replica, walker, ridx, collect(Θ+1:2Θ))
            sweep!_asymmetric(system, qmc, replica, walker, ridx, collect(Θ:-1:1))
        end

        jumpReplica && jump_replica!(replica, ridx)
        return nothing
    end
end

###################################################
##### Symmetric Sweep for Charge HS Transform #####
###################################################
function local_update!_symmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, ridx::Int,
    system::Hubbard, walker::HubbardWalker, replica::Replica;
    direction::Int=1, forceSymmetry::Bool=false,
    useHeatbath::Bool=true, saveRatio::Bool=true
)
    α = walker.α
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    ws = walker.ws

    σj = flip_HSField(σ[j])
    # compute ratios of determinants through G
    r, γ, ρ = compute_Metropolis_ratio(
        system, replica, walker, α[1, σj], j, ridx,
        direction=direction, forceSymmetry=forceSymmetry
    )
    saveRatio && push!(walker.tmp_r, r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)

    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1

        ### rank-1 updates ###
        # update imaginary time G
        update_Gτ0!(Gτ0, γ, Gτ, j, ws, direction=direction)
        update_G0τ!(G0τ, γ, Gτ, j, ws, direction=direction)
        # update Gτ, standard
        update_G!(Gτ, γ, 1.0, j, ws, direction=direction)
        # update Grover inverse
        update_invGA!(replica, ρ)
    end
end

function update_cluster!_symmetric(
    walker::HubbardWalker, replica::Replica,
    system::Hubbard, qmc::QMC, cidx::Int, ridx::Int;
    direction::Int=1
)
    k = qmc.K_interval[cidx]
    Θ = div(qmc.K, 2)

    direction == 1 ? (
        # propagate from τ to τ+k
        Bk = system.Bk;
        Bk⁻¹ = system.Bk⁻¹;
        slice = collect(1:k);
        Bc = walker.Bc.B[cidx-Θ]
    ) :
    (
        # propagate from τ+k to τ
        Bk = system.Bk⁻¹;
        Bk⁻¹ = system.Bk;
        slice = collect(k:-1:1);
        Bc = walker.Bc.B[cidx]
    )

    # all Green's functions
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0_up
    g0τ = walker.g0τ_up

    ws = walker.ws
    Bl = walker.Bl.B

    for i in slice
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || begin
            wrap_G!(Gτ, Bk, Bk⁻¹, ws)
            wrap_G!(Gτ0, Bk, Bk⁻¹, ws)
            wrap_G!(G0τ, Bk, Bk⁻¹, ws)
        end

        for j in 1:system.V
            local_update!_symmetric(σ, j, l, ridx,
                system, walker, replica,
                direction=direction,
                forceSymmetry=qmc.forceSymmetry,
                saveRatio=qmc.saveRatio,
                useHeatbath=qmc.useHeatbath
            )
        end

        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || begin
            wrap_G!(Gτ, Bk⁻¹, Bk, ws)
            wrap_G!(Gτ0, Bk⁻¹, Bk, ws)
            wrap_G!(G0τ, Bk⁻¹, Bk, ws)
        end

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], σ, system, tmpmat=ws.M)

        ### proceed to next time slice ###
        wrap_Gs!(Gτ, Gτ0, G0τ, Bl[i], ws, direction=direction)
    end

    @views prod_cluster!(Bc, Bl[k:-1:1], ws.M)
    proceed_gτ0!(gτ0[cidx], Bc, Gτ, ws, direction=direction)
    proceed_g0τ!(g0τ[cidx], Bc, Gτ, ws, direction=direction)

    return nothing
end

function sweep!_symmetric(
    system::Hubbard, qmc::QMC,
    replica::Replica, walker::HubbardWalker,
    ridx::Int, slice::Vector{Int}
)
    direction = slice[1] < slice[end] ? 1 : 2
    ### set alias ###
    Θ = div(qmc.K, 2)
    Aidx = replica.Aidx
    ws = walker.ws
    logdetGA, sgnlogdetGA = replica.logdetGA_up, replica.sgnlogdetGA_up
    φ₀ = walker.φ₀[1]
    φ₀T = walker.φ₀T[1]
    # temporal factorizations
    Fτt, FτT, _ = walker.Fτ
    Fl = walker.Fl[1]
    Fr = walker.Fr[1]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B
    # imaginary-time-displaced Green's
    Gτ = walker.G[1]
    Gτ0 = walker.Gτ0[1]
    G0τ = walker.G0τ[1]
    gτ0 = walker.gτ0_up
    g0τ = walker.g0τ_up

    Ul = walker.Ul_up
    Ur = walker.Ur_up

    ridx == 1 ? (G₀ = replica.G₀1_up;
    G₀′ = replica.G₀2_up) :
    (G₀ = replica.G₀2_up;
    G₀′ = replica.G₀1_up)

    # propagate from θ to 2θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_symmetric(walker, replica, system, qmc, cidx, ridx, direction=1)

            # multiply the updated slice to the right factorization on the left
            Bc = walker.Bc.B[cidx-Θ]
            lmul!(Bc, Fτt, ws)

            # Gτ needs to be periodically recomputed from scratch
            mul!(FτT, Fτt, Fr, ws)
            compute_G!(Gτ, φ₀, φ₀T, Ul, Ur, Fcl[i], FτT)

            # recompute imaginary-time-displaced Green's
            @views prod_cluster!(Gτ0, gτ0[cidx:-1:Θ+1], ws.M)
            @views prod_cluster!(G0τ, g0τ[Θ+1:cidx], ws.M)
            (cidx - Θ - 1) % 2 == 0 || @. G0τ *= -1

            # recompute G₀
            mul!(FτT, Fcl[i], Fτt, ws)
            compute_G!(G₀, φ₀, φ₀T, Ul, Ur, FτT, Fr)

            # recompute Grover inverse
            ridx == 1 ? begin
                logdetGA[], sgnlogdetGA[] = @views inv_Grover!(replica.GA⁻¹_up, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
            end :
            begin
                logdetGA[], sgnlogdetGA[] = @views inv_Grover!(replica.GA⁻¹_up, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
            end

            cidx == 2Θ && (
                copyto!(Fl, FτT);
                copyto!(Gτ, G₀);
                copyto!(Gτ0, Gτ);
                copyto!(G0τ, Gτ);
                G0τ[diagind(G0τ)] .-= 1
            )
        end

        # recompute all partial factorizations
        build_propagator!(Fcl, walker.Bc, ws, K=Θ, isReverse=true, isSymmetric=true)
        # reset temporal factorizations
        ldr!(Fτt, I)
        ldr!(FτT, I)

        # copy green's function to the spin-down sector
        copyto!(walker.G[2], walker.G[1])
        qmc.forceSymmetry && conj!(walker.G[2])

        return nothing
    end

    # propagate from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_symmetric(walker, replica, system, qmc, cidx, ridx, direction=2)

        # multiply the updated slice to the left factorization on the right
        Bc = walker.Bc.B[cidx]
        rmul!(Fτt, Bc, ws)

        # G needs to be periodically recomputed from scratch
        mul!(FτT, Fl, Fτt, ws)
        compute_G!(Gτ, φ₀, φ₀T, Ul, Ur, FτT, Fcr[i])

        # recompute imaginary-time-displaced Green's
        @views prod_cluster!(Gτ0, gτ0[Θ:-1:cidx], ws.M)
        @views prod_cluster!(G0τ, g0τ[cidx:Θ], ws.M)
        (Θ - cidx) % 2 == 0 || @. G0τ *= -1

        # recompute G₀
        mul!(FτT, Fτt, Fcr[i], ws)
        compute_G!(G₀, φ₀, φ₀T, Ul, Ur, Fl, FτT)

        # recompute Grover inverse
        ridx == 1 ? begin
            logdetGA[], sgnlogdetGA[] = @views inv_Grover!(replica.GA⁻¹_up, G₀[Aidx, Aidx], G₀′[Aidx, Aidx], replica.ws)
        end :
        begin
            logdetGA[], sgnlogdetGA[] = @views inv_Grover!(replica.GA⁻¹_up, G₀′[Aidx, Aidx], G₀[Aidx, Aidx], replica.ws)
        end

        cidx == 1 && (
            copyto!(Fr, FτT);
            copyto!(Gτ, G₀);
            copyto!(Gτ0, Gτ);
            copyto!(G0τ, Gτ);
            G0τ[diagind(G0τ)] .-= 1
        )
    end

    # recompute all partial factorizations
    build_propagator!(Fcr, walker.Bc, ws, K=Θ, isReverse=false, isSymmetric=true)
    # reset temporal factorizations
    ldr!(Fτt, I)
    ldr!(FτT, I)

    # copy green's function to the spin-down sector
    copyto!(walker.G[2], walker.G[1])
    qmc.forceSymmetry && conj!(walker.G[2])

    return nothing
end

function jump_replica!(replica::Replica, ridx::Int)
    Aidx = replica.Aidx
    G₀_up = ridx == 1 ? replica.G₀1_up : replica.G₀2_up
    G₀_dn = ridx == 1 ? replica.G₀1_dn : replica.G₀2_dn
    Im2GA_up = replica.Im2GA_up
    Im2GA_dn = replica.Im2GA_dn

    @views G′_up = G₀_up[Aidx, Aidx]
    @views G′_dn = G₀_dn[Aidx, Aidx]
    for i in CartesianIndices(Im2GA_up)
        @inbounds Im2GA_up[i] = -2 * G′_up[i]
    end
    for i in CartesianIndices(Im2GA_dn)
        @inbounds Im2GA_dn[i] = -2 * G′_dn[i]
    end
    Im2GA_up[diagind(Im2GA_up)] .+= 1
    Im2GA_dn[diagind(Im2GA_dn)] .+= 1

    return replica
end


#-------------------------------------------------------------------------------

function local_update!_asymmetric(
    σ::AbstractArray{Int}, j::Int, l::Int, ridx::Int,
    system::Hubbard, walker::HubbardWalker, replica::Replica;
    direction::Int=1,
    useHeatbath::Bool=true, saveRatio::Bool=true
)
    α = walker.α # 2x2 matrix relating to auxiliary field and HS transform
    Gτ_up = walker.G[1]
    Gτ0_up = walker.Gτ0[1]
    G0τ_up = walker.G0τ[1]
    Gτ_dn = walker.G[2]
    Gτ0_dn = walker.Gτ0[2]
    G0τ_dn = walker.G0τ[2]
    ws = walker.ws

    σj = flip_HSField(σ[j])

    r, γ_up, γ_dn, ρ_up, ρ_dn = compute_Metropolis_ratio_asymmetric(
        system, replica, walker, α, σj, j, ridx,
        direction=direction
    )
    saveRatio && push!(walker.tmp_r, r)
    # println(r)
    r = abs(r)
    # accept ratio
    u = useHeatbath ? real(r) / (1 + real(r)) : real(r)

    if rand() < u
        # accept the move, update the field and the Green's function
        walker.auxfield[j, l] *= -1

        ### rank-1 updates ###
        # update imaginary time G
        update_Gτ0!(Gτ0_up, γ_up, Gτ_up, j, ws, direction=direction)
        update_Gτ0!(Gτ0_dn, γ_dn, Gτ_dn, j, ws, direction=direction)

        update_G0τ!(G0τ_up, γ_up, Gτ_up, j, ws, direction=direction)
        update_G0τ!(G0τ_dn, γ_dn, Gτ_dn, j, ws, direction=direction)

        update_G!(Gτ_up, γ_up, 1.0, j, ws, direction=direction)
        update_G!(Gτ_dn, γ_dn, 1.0, j, ws, direction=direction)

        update_invGA!(replica, ρ_up, 1)
        update_invGA!(replica, ρ_dn, 2)
    end
end

function update_cluster!_asymmetric(
    walker::HubbardWalker, replica::Replica,
    system::Hubbard, qmc::QMC, cidx::Int, ridx::Int;
    direction::Int=1
)
    k = qmc.K_interval[cidx]
    Θ = div(qmc.K, 2) # K is the number of time slices divided by 

    direction == 1 ? (
        # propagate from τ to τ+k
        Bk = system.Bk;
        Bk⁻¹ = system.Bk⁻¹;
        slice = collect(1:k);
        Bc_up = walker.Bc.B[cidx-Θ];
        Bc_dn = walker.Bc.B[cidx]
    ) :
    (
        # propagate from τ+k to τ
        Bk = system.Bk⁻¹;
        Bk⁻¹ = system.Bk;
        slice = collect(k:-1:1);
        Bc_up = walker.Bc.B[cidx];
        Bc_dn = walker.Bc.B[cidx+Θ]
    )

    Gτ_up = walker.G[1]
    Gτ0_up = walker.Gτ0[1]
    G0τ_up = walker.G0τ[1]
    gτ0_up = walker.gτ0_up
    g0τ_up = walker.g0τ_up

    Gτ_dn = walker.G[2]
    Gτ0_dn = walker.Gτ0[2]
    G0τ_dn = walker.G0τ[2]
    gτ0_dn = walker.gτ0_dn
    g0τ_dn = walker.g0τ_dn

    ws = walker.ws
    Bl = walker.Bl.B

    for i in slice
        l = (cidx - 1) * qmc.stab_interval + i
        @views σ = walker.auxfield[:, l]

        # compute G <- Bk * G * Bk⁻¹ to enable fast update
        system.useFirstOrderTrotter || begin
            wrap_G!(Gτ_up, Bk, Bk⁻¹, ws)
            wrap_G!(Gτ0_up, Bk, Bk⁻¹, ws)
            wrap_G!(G0τ_up, Bk, Bk⁻¹, ws)

            wrap_G!(Gτ_dn, Bk, Bk⁻¹, ws)
            wrap_G!(Gτ0_dn, Bk, Bk⁻¹, ws)
            wrap_G!(G0τ_dn, Bk, Bk⁻¹, ws)
        end

        for j in 1:system.V
            local_update!_asymmetric(σ, j, l, ridx,
                system, walker, replica,
                direction=direction,
                saveRatio=qmc.saveRatio,
                useHeatbath=qmc.useHeatbath
            )
        end

        # compute G <- Bk⁻¹ * G * Bk to restore the ordering
        system.useFirstOrderTrotter || begin
            wrap_G!(Gτ_up, Bk⁻¹, Bk, ws)
            wrap_G!(Gτ0_up, Bk⁻¹, Bk, ws)
            wrap_G!(G0τ_up, Bk⁻¹, Bk, ws)

            wrap_G!(Gτ_dn, Bk⁻¹, Bk, ws)
            wrap_G!(Gτ0_dn, Bk⁻¹, Bk, ws)
            wrap_G!(G0τ_dn, Bk⁻¹, Bk, ws)
        end

        @views σ = walker.auxfield[:, l]
        imagtime_propagator!(Bl[i], Bl[k+i], σ, system, tmpmat=ws.M)

        ### proceed to next time slice ###
        wrap_Gs!(Gτ_up, Gτ0_up, G0τ_up, Bl[i], ws, direction=direction)
        wrap_Gs!(Gτ_dn, Gτ0_dn, G0τ_dn, Bl[k+i], ws, direction=direction)
    end

    @views prod_cluster!(Bc_up, Bl[k:-1:1], ws.M)
    @views prod_cluster!(Bc_dn, Bl[2k:-1:k+1], ws.M)

    proceed_gτ0!(gτ0_up[cidx], Bc_up, Gτ_up, ws, direction=direction)
    proceed_g0τ!(g0τ_up[cidx], Bc_up, Gτ_up, ws, direction=direction)

    proceed_gτ0!(gτ0_dn[cidx], Bc_dn, Gτ_dn, ws, direction=direction)
    proceed_g0τ!(g0τ_dn[cidx], Bc_dn, Gτ_dn, ws, direction=direction)

    return nothing
end

function sweep!_asymmetric(
    system::Hubbard, qmc::QMC,
    replica::Replica, walker::HubbardWalker,
    ridx::Int, slice::Vector{Int}
)
    direction = slice[1] < slice[end] ? 1 : 2
    ### set alias ###
    Θ = div(qmc.K, 2)
    Aidx = replica.Aidx
    ws = walker.ws

    logdetGA_up, sgnlogdetGA_up = replica.logdetGA_up, replica.sgnlogdetGA_up
    logdetGA_dn, sgnlogdetGA_dn = replica.logdetGA_dn, replica.sgnlogdetGA_dn
    φ₀_up = walker.φ₀[1]
    φ₀T_up = walker.φ₀T[1]
    φ₀_dn = walker.φ₀[2]
    φ₀T_dn = walker.φ₀T[2]

    # temporal factorizations
    Fτt_up, FτT_up, Fτt_dn, FτT_dn = walker.Fτ
    Fl_up = walker.Fl[1]
    Fr_up = walker.Fr[1]
    Fl_dn = walker.Fl[2]
    Fr_dn = walker.Fr[2]
    Fcl = walker.Fcl.B
    Fcr = walker.Fcr.B

    # imaginary-time-displaced Green's
    Gτ_up = walker.G[1]
    Gτ0_up = walker.Gτ0[1]
    G0τ_up = walker.G0τ[1]
    gτ0_up = walker.gτ0_up
    g0τ_up = walker.g0τ_up

    Gτ_dn = walker.G[2]
    Gτ0_dn = walker.Gτ0[2]
    G0τ_dn = walker.G0τ[2]
    gτ0_dn = walker.gτ0_dn
    g0τ_dn = walker.g0τ_dn

    Ul_up = walker.Ul_up
    Ur_up = walker.Ur_up

    Ul_dn = walker.Ul_dn
    Ur_dn = walker.Ur_dn

    ridx == 1 ? (G₀_up = replica.G₀1_up;
    G₀′_up = replica.G₀2_up;
    G₀_dn = replica.G₀1_dn;
    G₀′_dn = replica.G₀2_dn) :
    (G₀_up = replica.G₀2_up;
    G₀′_up = replica.G₀1_up;
    G₀_dn = replica.G₀2_dn;
    G₀′_dn = replica.G₀1_dn)

    # propagate from θ to 2θ
    direction == 1 && begin
        for (i, cidx) in enumerate(slice)
            update_cluster!_asymmetric(walker, replica, system, qmc, cidx, ridx, direction=1)

            # multiply the updated slice to the right factorization on the left
            Bc_up = walker.Bc.B[cidx-Θ]
            Bc_dn = walker.Bc.B[cidx]
            lmul!(Bc_up, Fτt_up, ws)
            lmul!(Bc_dn, Fτt_dn, ws)

            # Gτ needs to be periodically recomputed from scratch
            mul!(FτT_up, Fτt_up, Fr_up, ws)
            mul!(FτT_dn, Fτt_dn, Fr_dn, ws)
            compute_G!(Gτ_up, φ₀_up, φ₀T_up, Ul_up, Ur_up, Fcl[i], FτT_up)
            compute_G!(Gτ_dn, φ₀_dn, φ₀T_dn, Ul_dn, Ur_dn, Fcl[i+Θ], FτT_dn)

            # recompute imaginary-time-displaced Green's
            @views prod_cluster!(Gτ0_up, gτ0_up[cidx:-1:Θ+1], ws.M)
            @views prod_cluster!(Gτ0_dn, gτ0_dn[cidx:-1:Θ+1], ws.M)
            @views prod_cluster!(G0τ_up, g0τ_up[Θ+1:cidx], ws.M)
            @views prod_cluster!(G0τ_dn, g0τ_dn[Θ+1:cidx], ws.M)
            (cidx - Θ - 1) % 2 == 0 || @. G0τ_up *= -1
            (cidx - Θ - 1) % 2 == 0 || @. G0τ_dn *= -1

            # recompute G₀
            mul!(FτT_up, Fcl[i], Fτt_up, ws)
            mul!(FτT_dn, Fcl[i+Θ], Fτt_dn, ws)
            compute_G!(G₀_up, φ₀_up, φ₀T_up, Ul_up, Ur_up, FτT_up, Fr_up)
            compute_G!(G₀_dn, φ₀_dn, φ₀T_dn, Ul_dn, Ur_dn, FτT_dn, Fr_dn)

            # recompute Grover inverse
            ridx == 1 ? begin
                logdetGA_up[], sgnlogdetGA_up[] = @views inv_Grover!(replica.GA⁻¹_up, G₀_up[Aidx, Aidx], G₀′_up[Aidx, Aidx], replica.ws)
                logdetGA_dn[], sgnlogdetGA_dn[] = @views inv_Grover!(replica.GA⁻¹_dn, G₀_dn[Aidx, Aidx], G₀′_dn[Aidx, Aidx], replica.ws)
            end :
            begin
                logdetGA_up[], sgnlogdetGA_up[] = @views inv_Grover!(replica.GA⁻¹_up, G₀′_up[Aidx, Aidx], G₀_up[Aidx, Aidx], replica.ws)
                logdetGA_dn[], sgnlogdetGA_dn[] = @views inv_Grover!(replica.GA⁻¹_dn, G₀′_dn[Aidx, Aidx], G₀_dn[Aidx, Aidx], replica.ws)
            end

            cidx == 2Θ && (
                copyto!(Fl_up, FτT_up);
                copyto!(Fl_dn, FτT_dn);
                copyto!(Gτ_up, G₀_up);
                copyto!(Gτ_dn, G₀_dn);
                copyto!(Gτ0_up, Gτ_up);
                copyto!(Gτ0_dn, Gτ_dn);
                copyto!(G0τ_up, Gτ_up);
                copyto!(G0τ_dn, Gτ_dn);
                G0τ_up[diagind(G0τ_up)] .-= 1;
                G0τ_dn[diagind(G0τ_dn)] .-= 1
            )
        end

        # recompute all partial factorizations
        build_propagator!(Fcl, walker.Bc, ws, K=Θ, isReverse=true, isSymmetric=false)
        # reset temporal factorizations
        ldr!(Fτt_up, I)
        ldr!(Fτt_dn, I)
        ldr!(FτT_up, I)
        ldr!(FτT_dn, I)

        return nothing
    end

    # propagate from θ to 0
    for (i, cidx) in zip(Iterators.reverse(eachindex(slice)), slice)
        update_cluster!_asymmetric(walker, replica, system, qmc, cidx, ridx, direction=2)

        # multiply the updated slice to the left factorization on the right
        Bc_up = walker.Bc.B[cidx]
        Bc_dn = walker.Bc.B[cidx+Θ]
        rmul!(Fτt_up, Bc_up, ws)
        rmul!(Fτt_dn, Bc_dn, ws)

        # G needs to be periodically recomputed from scratch
        mul!(FτT_up, Fl_up, Fτt_up, ws)
        mul!(FτT_dn, Fl_dn, Fτt_dn, ws)
        compute_G!(Gτ_up, φ₀_up, φ₀T_up, Ul_up, Ur_up, FτT_up, Fcr[i])
        compute_G!(Gτ_dn, φ₀_dn, φ₀T_dn, Ul_dn, Ur_dn, FτT_dn, Fcr[i+Θ])

        # recompute imaginary-time-displaced Green's
        @views prod_cluster!(Gτ0_up, gτ0_up[Θ:-1:cidx], ws.M)
        @views prod_cluster!(Gτ0_dn, gτ0_dn[Θ:-1:cidx], ws.M)
        @views prod_cluster!(G0τ_up, g0τ_up[cidx:Θ], ws.M)
        @views prod_cluster!(G0τ_dn, g0τ_dn[cidx:Θ], ws.M)
        (Θ - cidx) % 2 == 0 || @. G0τ_up *= -1
        (Θ - cidx) % 2 == 0 || @. G0τ_dn *= -1

        # recompute G₀
        mul!(FτT_up, Fτt_up, Fcr[i], ws)
        mul!(FτT_dn, Fτt_dn, Fcr[i+Θ], ws)
        compute_G!(G₀_up, φ₀_up, φ₀T_up, Ul_up, Ur_up, Fl_up, FτT_up)
        compute_G!(G₀_dn, φ₀_dn, φ₀T_dn, Ul_dn, Ur_dn, Fl_dn, FτT_dn)

        # recompute Grover inverse
        ridx == 1 ? begin
            logdetGA_up[], sgnlogdetGA_up[] = @views inv_Grover!(replica.GA⁻¹_up, G₀_up[Aidx, Aidx], G₀′_up[Aidx, Aidx], replica.ws)
            logdetGA_dn[], sgnlogdetGA_dn[] = @views inv_Grover!(replica.GA⁻¹_dn, G₀_dn[Aidx, Aidx], G₀′_dn[Aidx, Aidx], replica.ws)
        end :
        begin
            logdetGA_up[], sgnlogdetGA_up[] = @views inv_Grover!(replica.GA⁻¹_up, G₀′_up[Aidx, Aidx], G₀_up[Aidx, Aidx], replica.ws)
            logdetGA_dn[], sgnlogdetGA_dn[] = @views inv_Grover!(replica.GA⁻¹_dn, G₀′_dn[Aidx, Aidx], G₀_dn[Aidx, Aidx], replica.ws)
        end

        cidx == 1 && (
            copyto!(Fr_up, FτT_up);
            copyto!(Fr_dn, FτT_dn);
            copyto!(Gτ_up, G₀_up);
            copyto!(Gτ_dn, G₀_dn);
            copyto!(Gτ0_up, Gτ_up);
            copyto!(Gτ0_dn, Gτ_dn);
            copyto!(G0τ_up, Gτ_up);
            copyto!(G0τ_dn, Gτ_dn);
            G0τ_up[diagind(G0τ_up)] .-= 1;
            G0τ_dn[diagind(G0τ_dn)] .-= 1
        )
    end

    # recompute all partial factorizations
    build_propagator!(Fcr, walker.Bc, ws, K=Θ, isReverse=false, isSymmetric=false)
    # reset temporal factorizations
    ldr!(Fτt_up, I)
    ldr!(Fτt_dn, I)
    ldr!(FτT_up, I)
    ldr!(FτT_dn, I)

    return nothing
end