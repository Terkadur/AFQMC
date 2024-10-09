"""
    Replica{W, T, E}

Defines a replica with the given parameters.
"""
struct Replica{W, T, E}
    ### Partition indices ###
    Aidx::Vector{Int64}

    ### Two replica walkers ###
    walker1::W
    walker2::W

    ### Allocating Temporal Data ###
    # Green's functions at imaginary time θ/2
    G₀1_up::Matrix{T}
    G₀2_up::Matrix{T}
    G₀1_dn::Matrix{T}
    G₀2_dn::Matrix{T}
    
    # Inverse of the Grover matrix
    GA⁻¹_up::Matrix{T}
    GA⁻¹_dn::Matrix{T}
    logdetGA_up::Base.RefValue{Float64}    # note: it's the negative value log(detGA⁻¹)
    sgnlogdetGA_up::Base.RefValue{T}
    logdetGA_dn::Base.RefValue{Float64}    # note: it's the negative value log(detGA⁻¹)
    sgnlogdetGA_dn::Base.RefValue{T}

    # Matrix I - 2*GA, where GA is the submatrix of either G₀1 or G₀2, depending on which replica is currently NOT being updated
    Im2GA_up::Matrix{T}
    Im2GA_dn::Matrix{T}

    # Allocating three vectors used in computing the ratio and updating the Grover inverse
    a_up::Vector{T}
    b_up::Vector{T}
    t_up::Vector{T}
    a_dn::Vector{T}
    b_dn::Vector{T}
    t_dn::Vector{T}

    ### Thermaldynamic integration variable for incremental algorithm ###
    λₖ::Float64

    ### LDR Workspace ###
    ws::LDRWorkspace{T, E}

    function Replica(extsys::ExtendedSystem, w1::W, w2::W; λₖ::Float64 = 1.0) where W
        T = eltype(w1.G[1])

        LA = extsys.LA
        Aidx = extsys.Aidx
        GA⁻¹_up = zeros(T, LA, LA)
        GA⁻¹_dn = zeros(T, LA, LA)
        ws = ldr_workspace(GA⁻¹_up)
        G₀1_up, G₀2_up = copy(w1.G[1]), copy(w2.G[1])
        G₀1_dn, G₀2_dn = copy(w1.G[2]), copy(w2.G[2])
        logdetGA_up, sgnlogdetGA_up = inv_Grover!(GA⁻¹_up, G₀1_up[Aidx, Aidx], G₀2_up[Aidx, Aidx], ws)
        logdetGA_dn, sgnlogdetGA_dn = inv_Grover!(GA⁻¹_dn, G₀1_dn[Aidx, Aidx], G₀2_dn[Aidx, Aidx], ws)
        a_up, b_up, t_up = zeros(T, LA), zeros(T, LA), zeros(T, LA)
        a_dn, b_dn, t_dn = zeros(T, LA), zeros(T, LA), zeros(T, LA)
        Im2GA_up = I - 2 * G₀2_up[1:LA, 1:LA]
        Im2GA_dn = I - 2 * G₀2_dn[1:LA, 1:LA]

        return new{W, T, Float64}(
            Aidx, 
            w1, w2, 
            G₀1_up, G₀2_up, G₀1_dn, G₀2_dn,  
            GA⁻¹_up, GA⁻¹_dn, Ref(logdetGA_up), Ref(sgnlogdetGA_up), Ref(logdetGA_dn), Ref(sgnlogdetGA_dn),
            Im2GA_up, Im2GA_dn, a_up, b_up, t_up, a_dn, b_dn, t_dn, 
            λₖ, ws
        )
    end
end

### Display Info ###
Base.summary(r::Replica) = string(nameof(typeof(r)))

function Base.show(io::IO, r::Replica)
    println(io, TYPE_COLOR, Base.summary(r), NO_COLOR)
    println(io, "Partition size: ", TYPE_COLOR, length(r.Aidx), NO_COLOR)
    println(io, "log(detGA⁻¹): ", TYPE_COLOR, r.logdetGA[], NO_COLOR)
end

"""
    update!(r::Replica)

Updates the value of det(GA⁻¹) in the provided replica.
"""
function update!(r::Replica)
    Aidx, G₀1, G₀2, GA⁻¹, ws = r.Aidx, r.walker1.G[1], r.walker2.G[1], r.GA⁻¹, r.ws
    logdetGA, sgnlogdetGA = inv_Grover!(GA⁻¹, G₀1[Aidx, Aidx], G₀2[Aidx, Aidx], ws)
    r.logdetGA[] = logdetGA
    r.sgnlogdetGA[] = sgnlogdetGA
    copyto!(r.G₀1, G₀1)
    copyto!(r.G₀2, G₀2)

    return r
end
