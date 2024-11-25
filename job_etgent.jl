include("./qmc_etgent.jl")

seed = parse(Int64, ARGS[1])
@show seed
Random.seed!(seed)

const Nₖ = 5
const λₖ_list = collect(0.0:0.2:0.8)

const Lx, Ly = 4, 4
const T = hopping_matrix_Hubbard_2d(Lx, Ly, 1.0)
const U = 8.0
@show U

const N_up, N_dn = 8, 8 #parse(Int64, ARGS[2]), parse(Int64, ARGS[3])

const system = GenericHubbard(
    # (Nx, Ny), (N_up, N_dn)
    (Lx, Ly, 1), (N_up, N_dn),
    # t, U
    T, U,
    # μ
    0.0,
    # β, L
    20.0, 200,
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
    saveRatio=false
)


const φ₀_up = trial_wf_free(system, 1, T)
const φ₀_dn = trial_wf_free(system, 2, T)
const φ₀ = [φ₀_up, φ₀_dn]

const Aidx = collect(1:8)
const extsys = ExtendedSystem(system, Aidx, subsysOrdering=false)

path = "./data/4x4/"

swap_period = 256

for λₖ in λₖ_list
    # 6617s (1h50m)
    @show λₖ
    filename = "etgent/EtgEnt_LA$(length(Aidx))_N$(sum(system.N))_U$(system.U)_lambda$(λₖ)_beta$(system.β)_seed$(seed).jld"
    @time run_incremental_sampling_gs(extsys, qmc, φ₀, λₖ, Nₖ, path, filename, swap_period)
end
