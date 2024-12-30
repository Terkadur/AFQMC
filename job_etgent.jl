include("./qmc_etgent.jl")

seed = parse(Int64, ARGS[1])
@show seed
Random.seed!(seed)

const Nₖ = 1
const λₖ_list = collect(0.0:0.2:0.8)

const Lx, Ly = 2, 2
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)
const U = 2.0
@show U

const N_up, N_dn = 1, 2

const β = parse(Float64, ARGS[2])
@show β

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (N_up, N_dn),
    # t, U
    T, U,
    # μ
    0.0,
    # β, L
    β, 10 * convert(Int64, β),
    # data type of the systems
    sys_type=ComplexF64,
    # if use charge decomposition
    useChargeHST=false,
    # if use first-order Trotteriaztion
    useFirstOrderTrotter=false
)

const qmc = QMC(
    system,
    # number of warm-ups, samples and measurement interval
    1024, 8192, 10,
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
# const φ₀ = trial_wf_HF(system, ϵ=1e-10)
const φ₀ = trial_wf_free_asym(system, T)

const Aidx = collect(1:2)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

path = "./data_with_sgn/2x2"

swap_period = 256

const λₖ = 0.0 #parse(Float64, ARGS[3])
@show λₖ
filename = "etgent_Nk1/EtgEnt_LA$(length(Aidx))_Nup$(system.N[1])_Ndn$(system.N[2])_U$(system.U)_lambda$(λₖ)_beta$(system.β)_seed$(seed).jld"
@time run_incremental_sampling_gs(extsys, qmc, φ₀, λₖ, Nₖ, path, filename, swap_period)
