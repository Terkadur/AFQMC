
using JLD, Measurements, Statistics

# U_list1 = [-0.5, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
# U_list2 = [-7.0, -8.0]

S2_avg = zeros(Float64, 9)
# S2_err = zeros(Float64, 9)

for (i, U) in enumerate(U_list1)
    local file_list = ["./tarek_data/EtgEnt_LA16_N64_U$(U)_lambda$(λₖ)_beta50.0_seed1234.jld" for λₖ in collect(0.0:0.2:0.8)]
    local data = load.(file_list)

    local detgA_avg = [mean(data[j]["detgA"]) for j in collect(1:5)]
    local detgA_err = [std(data[j]["detgA"]) / length(data[j]["detgA"]) for j in collect(1:5)]

    local expmS2 = measurement.(detgA_avg, detgA_err)
    local S2 = -log(prod(expmS2))

    S2_avg[i] = S2.val
    S2_err[i] = S2.err

end

for (i, U) in enumerate(U_list2)
    local file_list = ["./tarek_data/EtgEnt_LA16_N64_U$(U)_lambda$(λₖ)_beta50.0_seed1235.jld" for λₖ in collect(0.0:0.2:0.8)]
    local data = load.(file_list)

    local detgA_avg = [mean(data[j]["detgA"]) for j in collect(1:5)]
    local detgA_err = [std(data[j]["detgA"]) / length(data[j]["detgA"]) for j in collect(1:5)]

    local expmS2 = measurement.(detgA_avg, detgA_err)
    local S2 = -log(prod(expmS2))

    S2_avg[i+length(U_list1)] = S2.val
    S2_err[i+length(U_list1)] = S2.err

end


jldopen("./tarek_data/processed/EtgEnt_LA16_N64_beta50.0_1seed.jld", "w") do file
    write(file, "U", cat(U_list1, U_list2; dims=1))
    write(file, "S2_avg", S2_avg)
    write(file, "S2_err", S2_err)
end