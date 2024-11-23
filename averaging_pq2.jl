
using JLD
include("./averaging_utils.jl")

path_src = "./data/2x2/pq2/"
path_dst = "./data/2x2/processed/"

# filling_list = vcat([(1, x) for x in 1:6],
#     [(2, x) for x in 2:6],
#     [(3, x) for x in 3:6],
#     [(4, x) for x in 4:6],
#     [(5, x) for x in 5:6],
#     [(6, 6)])
filling_list = [(1, 1), (1, 2), (2, 2), (3, 3)]
LA = 2

Pmn2_avg = zeros(Float64, LA + 1, LA + 1, length(filling_list))
Pmn2_err = zeros(Float64, LA + 1, LA + 1, length(filling_list))

Pn2_avg = zeros(Float64, 2 * LA + 1, length(filling_list))
Pn2_err = zeros(Float64, 2 * LA + 1, length(filling_list))
Pm2_avg = zeros(Float64, 2 * LA + 1, length(filling_list))
Pm2_err = zeros(Float64, 2 * LA + 1, length(filling_list))

Hn2_avg = zeros(Float64, length(filling_list))
Hn2_err = zeros(Float64, length(filling_list))
Hm2_avg = zeros(Float64, length(filling_list))
Hm2_err = zeros(Float64, length(filling_list))

possible_n = collect(0:2*LA)
possible_m = collect(-LA:LA)
En_conv = Vector{Float64}[]
Em_conv = Vector{Float64}[]

for (i, filling) in enumerate(filling_list)
    # merge data
    filelist = filter(x -> match(Regex("Pq2_LA$(LA)_Nup$(filling[1])_Ndn$(filling[2])_U2.0_beta18.0_*"), x) !== nothing, readdir(path_src))
    data = load.("$(path_src)" .* filelist)
    all_unmerged = get_probs_raw.(data)
    
    Pmn2_unmerged = [tup[1] for tup in all_unmerged]
    Pn2_unmerged = [tup[2] for tup in all_unmerged]
    Pm2_unmerged = [tup[3] for tup in all_unmerged]
    
    Pmn2_raw = cat(Pmn2_unmerged...; dims=3)
    Pn2_raw = cat(Pn2_unmerged...; dims=2)
    Pm2_raw = cat(Pm2_unmerged...; dims=2)
    
    Pmn2_filtered = reduce((x, y) -> cat(x, y; dims=3), [Pmn2_raw[:, :, i] for i=1:size(Pmn2_raw, 3) if !iszero(Pmn2_raw[:, :, i])])
    Pn2_filtered = reduce((x, y) -> cat(x, y; dims=2), [Pn2_raw[:, i] for i=1:size(Pn2_raw, 2) if !iszero(Pn2_raw[:, i])])
    Pm2_filtered = reduce((x, y) -> cat(x, y; dims=2), [Pm2_raw[:, i] for i=1:size(Pm2_raw, 2) if !iszero(Pm2_raw[:, i])])
    
    # stats
    Pmn2_avg[:, :, i], Pmn2_err[:, :, i], Pn2_avg[:, i], Pn2_err[:, i], Pm2_avg[:, i], Pm2_err[:, i] = get_probs_stats(Pmn2_filtered, Pn2_filtered, Pm2_filtered)
    Hn2_avg[i], Hn2_err[i], Hm2_avg[i], Hm2_err[i] = get_shannon_stats(Pn2_filtered, Pm2_filtered)

    # convergence
    En_raw = reshape(reshape(possible_n, 1, :) * Pn2_filtered, :)
    Em_raw = reshape(reshape(possible_m, 1, :) * Pm2_filtered, :)

    sweeps = length(En_raw)
    @assert length(Em_raw) == sweeps "Unequal number of sweeps for N and M"
    En_run = cumsum(En_raw) ./ collect(1:sweeps)
    Em_run = cumsum(Em_raw) ./ collect(1:sweeps)
    push!(En_conv, real.(En_run))
    push!(Em_conv, real.(Em_run))
end

jldopen("$(path_dst)" * "Pq2_LA$(LA)_U2.0_beta18.0.jld", "w") do file
    write(file, "filling", filling_list)
    write(file, "Pmn2_avg", Pmn2_avg)
    write(file, "Pmn2_err", Pmn2_err)
    write(file, "Pn2_avg", Pn2_avg)
    write(file, "Pn2_err", Pn2_err)
    write(file, "Pm2_avg", Pm2_avg)
    write(file, "Pm2_err", Pm2_err)
    write(file, "En_conv", En_conv)
    write(file, "Em_conv", Em_conv)
    write(file, "Hn2_avg", Hn2_avg)
    write(file, "Hn2_err", Hn2_err)
    write(file, "Hm2_avg", Hm2_avg)
    write(file, "Hm2_err", Hm2_err)
end