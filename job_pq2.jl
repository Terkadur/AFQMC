include("./qmc_pq2.jl")

const Lx, Ly = 4, 4
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)
const U = -1.0
@show U

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (8, 8),
    # t, U
    T, U,
    # μ
    0.0,
    # β, L
    18.0, 180,
    # data type of the system
    sys_type=ComplexF64,
    # if use charge decomposition
    useChargeHST=false,
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
    forceSymmetry=true,
    # debugging flag
    saveRatio=false
)

seed = 1234
@show seed
Random.seed!(seed)

const φ₀_up = trial_wf_free(system, 1, T)
const φ₀_dn = trial_wf_free(system, 2, T)
const φ₀ = [φ₀_up, φ₀_dn]

const Aidx = collect(1:4)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

path = "./data/"

filename = "Pq2_LA$(length(Aidx))_N$(sum(system.N))_U$(system.U)_beta$(system.β)_seed$(seed).jld"
@time run_regular_sampling_gs(extsys, qmc, φ₀, path, filename)