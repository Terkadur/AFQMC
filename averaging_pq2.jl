
using JLD
include("./averaging_utils.jl")

U_list = [-1.0]
LA = 4

Pmn2_avg = zeros(Float64, LA + 1, LA + 1, length(U_list))
Pmn2_err = zeros(Float64, LA + 1, LA + 1, length(U_list))

Pn2_avg = zeros(Float64, 2 * LA + 1, length(U_list))
Pn2_err = zeros(Float64, 2 * LA + 1, length(U_list))
Pm2_avg = zeros(Float64, 2 * LA + 1, length(U_list))
Pm2_err = zeros(Float64, 2 * LA + 1, length(U_list))

Hn2_avg = zeros(Float64, length(U_list))
Hn2_err = zeros(Float64, length(U_list))
Hm2_avg = zeros(Float64, length(U_list))
Hm2_err = zeros(Float64, length(U_list))

for (i, U) in enumerate(U_list)
    local file = "./data/Pq2_LA4_N16_U$(U)_beta18.0_seed1234.jld"
    local data = load(file)
    local Pmn2_raw, Pn2_raw, Pm2_raw = get_probs_raw(data)

    Pmn2_avg[:, :, i], Pmn2_err[:, :, i], Pn2_avg[:, i], Pn2_err[:, i], Pm2_avg[:, i], Pm2_err[:, i] = get_probs_stats(Pmn2_raw, Pn2_raw, Pm2_raw)
    Hn2_avg[i], Hn2_err[i], Hm2_avg[i], Hm2_err[i] = get_shannon_stats(Pn2_raw, Pm2_raw)
    # print(Pn2_avg)
end

jldopen("./data/Pq2_LA4_N16_beta18.0.jld", "w") do file
    write(file, "U", U_list)
    write(file, "Pmn2_avg", Pmn2_avg)
    write(file, "Pmn2_err", Pmn2_err)
    write(file, "Pn2_avg", Pn2_avg)
    write(file, "Pn2_err", Pn2_err)
    write(file, "Pm2_avg", Pm2_avg)
    write(file, "Pm2_err", Pm2_err)
end

# jldopen("./tarek_data/asymm/processed/Hq2_Lx4_Ly4_LA4_N12_beta18.0.jld", "w") do file
#     write(file, "U", U_list)
#     write(file, "Hn2_avg", Hn2_avg)
#     write(file, "Hn2_err", Hn2_err)
#     write(file, "Hm2_avg", Hm2_avg)
#     write(file, "Hm2_err", Hm2_err)
# end