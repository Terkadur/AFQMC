
using JLD, Measurements, Statistics

path = "./data/2x2/"
U_list = [2.0]
lambda_list = collect(0.0:0.2:0.8)
lambdas = length(lambda_list)

S2_avg = zeros(Float64, length(U_list))
S2_err = zeros(Float64, length(U_list))

S2_conv = Vector{Float64}[]

for (i, U) in enumerate(U_list)
    # merge data
    filelist = [filter(x -> match(Regex("EtgEnt_LA2_Nup1_Ndn3_U$(U)_lambda$(lambda)_beta50.0_*"), x) !== nothing, readdir(path)) for lambda in lambda_list]
    detgA_list = Vector{Float64}[]
    for filenames in filelist
        # raw data is saved in multiple files
        data = load.("$(path)" .* filenames)
        # merge raw data
        tmp = [data[i]["detgA"] for i in 1:length(filenames)]
        push!(detgA_list, vcat(tmp...))
    end

    # stats
    detgA_avg = mean.(detgA_list)
    detgA_err = std.(detgA_list) ./ length(detgA_list)

    expmS2 = measurement.(detgA_avg, detgA_err)
    S2 = -log(prod(expmS2))
    S2_avg[i] = S2.val
    S2_err[i] = S2.err

    # convergence
    sweeps = length(detgA_list[1])
    for detgAs in detgA_list
        @assert length(detgAs) == sweeps "Unequal number of sweeps for each lambda"
    end
    detgA_run = zeros(Float64, lambdas, sweeps)
    for (i, detgAs) in enumerate(detgA_list)
        detgA_run[i, :] = cumsum(detgAs) ./ collect(1:sweeps)
    end
    push!(S2_conv, vec(-log.(prod(detgA_run, dims=1))))
end

jldopen("$(path)" * "processed/EtgEnt_LA2_Nup1_Ndn3_beta50.0.jld", "w") do file
    write(file, "U", U_list)
    write(file, "S2_avg", S2_avg)
    write(file, "S2_err", S2_err)
    write(file, "S2_conv", S2_conv)
end