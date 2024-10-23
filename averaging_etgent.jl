
using JLD, Measurements, Statistics

U_list = collect(-8.0:1.0:8.0)

S2_avg = zeros(Float64, length(U_list))
S2_err = zeros(Float64, length(U_list))

for (i, U) in enumerate(U_list)
    local file_list = ["./data/EtgEnt_LA1_N2_U$(U)_lambda$(λₖ)_beta50.0_seed1234.jld" for λₖ in collect(0.0:0.2:0.8)]
    local data = load.(file_list)

    local detgA_avg = [mean(data[j]["detgA"]) for j in collect(1:5)]
    local detgA_err = [std(data[j]["detgA"]) / length(data[j]["detgA"]) for j in collect(1:5)]

    local expmS2 = measurement.(detgA_avg, detgA_err)
    local S2 = -log(prod(expmS2))

    S2_avg[i] = S2.val
    S2_err[i] = S2.err
end

jldopen("./data/processed/EtgEnt_LA1_N2_beta50.0_seed1234.jld", "w") do file
    write(file, "U", U_list)
    write(file, "S2_avg", S2_avg)
    write(file, "S2_err", S2_err)
end