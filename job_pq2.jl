include("./qmc_pq2.jl")

seed = 1234
@show seed
Random.seed!(seed)

const Lx, Ly = 4, 4
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)
const U = 2.0
@show U

const N_up, N_dn = 1, 2 #parse(Int64, ARGS[1]), parse(Int64, ARGS[2])

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (N_up, N_dn),
    # t, U
    T, U,
    # μ
    0.0,
    # β, L
    6.0, 60,
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
    1024, 16384, 10,
    # stablization and update interval
    10, 10,
    # if force spin symmetry
    forceSymmetry=false,
    # debugging flag
    saveRatio=false
)

# const φ₀_up = trial_wf_free(system, 1, T)
# const φ₀_dn = trial_wf_free(system, 2, T)
# const φ₀ = [φ₀_up, φ₀_dn]
# # const φ₀ = trial_wf_HF(system, ϵ=1e-10)
const φ₀ = trial_wf_free_asym(system, T)

const Aidx = collect(1:8)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

swap_period = 256

path = "./attr_data/3x3/sym"

filename_pq = "pq2/testPq2_LA$(length(Aidx))_Nup$(system.N[1])_Ndn$(system.N[2])_U$(system.U)_beta$(system.β)_seed$(seed).jld"
filename_sgn = "sgn/testSgn_LA$(length(Aidx))_Nup$(system.N[1])_Ndn$(system.N[2])_U$(system.U)_beta$(system.β)_seed$(seed).jld"
@time run_regular_sampling_gs(extsys, qmc, φ₀, path, filename_pq, filename_sgn, swap_period)
