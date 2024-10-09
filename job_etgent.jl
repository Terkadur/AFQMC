include("./qmc_etgent.jl")

const Nₖ = 5
const λₖ_list = collect(0.0:0.2:0.8)

const Lx, Ly = 4, 4
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)
const U = -1.0 # -0.5, -1, -2, -3, -4, -5, -6, -7, -8
@show U

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (8, 8),
    # t, U
    T, U,
    # μ
    0.0,
    # β, L
    50.0, 500,
    # data type of the system
    sys_type=ComplexF64,
    # if use charge decomposition
    useChargeHST=true,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

const qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    16, 32, 6,
    # stablization and update interval
    10, 10,
    # if force spin symmetry
    forceSymmetry=false,
    # debugging flag
    saveRatio=false
)

seed = 1234
@show seed
Random.seed!(seed)

const φ₀_up = trial_wf_free(system, 1, T)
const φ₀ = [φ₀_up, copy(φ₀_up)]

const Aidx = collect(1:4)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

path = "./data/"

# for λₖ in λₖ_list
#     # 6617s (1h50m)
#     @show λₖ
λₖ = 0.0
filename = "EtgEnt_LA$(length(Aidx))_N$(sum(system.N))_U$(system.U)_lambda$(λₖ)_beta$(system.β)_seed$(seed).jld"
@time run_incremental_sampling_gs(extsys, qmc, φ₀, λₖ, Nₖ, path, filename)
# end