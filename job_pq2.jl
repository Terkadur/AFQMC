include("./qmc_pq2.jl")

seed = 1234
@show seed
Random.seed!(seed)

const Lx, Ly = 2, 2
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)
const U = 2.0
@show U

const N_up, N_dn = parse(Int64, ARGS[1]), parse(Int64, ARGS[2])

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (N_up, N_dn),
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
    512, 8192, 10,
    # stablization and update interval
    10, 10,
    # if force spin symmetry
    forceSymmetry=false,
    # debugging flag
    saveRatio=true
)

const φ₀_up = trial_wf_free(system, 1, T)
const φ₀_dn = trial_wf_free(system, 2, T)
const φ₀ = [φ₀_up, φ₀_dn]

const Aidx = collect(1:2)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

swap_period = 256

path = "./data/2x2/"

filename_pq = "pq2/Pq2_LA$(length(Aidx))_N$(sum(system.N))_U$(system.U)_beta$(system.β)_seed$(seed).jld"
filename_sgn = "sgn/Sgn_LA$(length(Aidx))_N$(sum(system.N))_U$(system.U)_beta$(system.β)_seed$(seed).jld"
@time run_regular_sampling_gs(extsys, qmc, φ₀, path, filename_pq, filename_sgn, swap_period)