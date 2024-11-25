
using JLD, Measurements, Statistics

path_src = "./data/4x4/etgent/"
path_dst = "./data/4x4/processed/"
lambda_list = collect(0.0:0.2:0.8)
lambdas = length(lambda_list)

filling_list = vcat([(x, x) for x in collect(1:7)])
partition_list = [8] #collect(1:9)

S2_avg = zeros(Float64, length(partition_list))
S2_err = zeros(Float64, length(partition_list))

S2_conv = Vector{Float64}[]

for (i, partition) in enumerate(partition_list)
    # merge data
    filelist = [filter(x -> match(Regex("EtgEnt_LA$(partition)_N16_U8.0_lambda$(lambda)_beta20.0_*"), x) !== nothing, readdir(path_src)) for lambda in lambda_list]
    if isempty(filelist[1])
        throw("no files found")
    end
    detgA_list = Vector{Float64}[]
    for filenames in filelist
        # raw data is saved in multiple files
        data = load.("$(path_src)" .* filenames)
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

jldopen("$(path_dst)" * "EtgEnt_N16_U8.0_beta20.0.jld", "w") do file
    write(file, "partition", partition_list)
    write(file, "S2_avg", S2_avg)
    write(file, "S2_err", S2_err)
    write(file, "S2_conv", S2_conv)
end