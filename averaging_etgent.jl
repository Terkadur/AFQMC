
using JLD, Measurements, Statistics

path_src = "./rep_data/2x2/etgent_with_sgn/"
path_dst = "./rep_data/2x2/processed/"
lambda_list = collect(0.0:0.2:0.8)
lambdas = length(lambda_list)

# filling_list = [(2, 6)]
# partition_list = [8] #collect(1:9)

S2_avg = zeros(Float64, 1)
S2_err = zeros(Float64, 1)

S2_conv = Vector{Float64}[]

# for (i, filling) in enumerate(filling_list)
i = 1
filling = (2, 2)
# merge data
filelist = [filter(x -> match(Regex("EtgEnt_LA2_Nup$(filling[1])_Ndn$(filling[2])_U2.0_lambda$(lambda)_beta50.0_*"), x) !== nothing, readdir(path_src)) for lambda in lambda_list]
if isempty(filelist[1])
    throw("no files found")
end
detgA_list = Vector{Float64}[]
sgn_list = Vector{Float64}[]
for filenames in filelist
    # raw data is saved in multiple files
    data = load.("$(path_src)" .* filenames)
    # merge raw data
    tmp = [data[i]["detgA"] for i in 1:length(filenames)]
    push!(detgA_list, vcat(tmp...))

    tmp = [data[i]["sgn"] for i in 1:length(filenames)]
    push!(sgn_list, vcat(tmp...))
end

# stats
detgA_avg = mean.(detgA_list)
detgA_err = std.(detgA_list) ./ length(detgA_list)
expmS2 = measurement.(detgA_avg, detgA_err)
S2 = -log(prod(expmS2))
S2_avg[i] = S2.val
S2_err[i] = S2.err

@show S2.val

# convergence
# sweeps = length(detgA_list[1])
# for detgAs in detgA_list
#     @assert length(detgAs) == sweeps "Unequal number of sweeps for each lambda"
# end
# detgA_run = zeros(Float64, lambdas, sweeps)
# for (i, detgAs) in enumerate(detgA_list)
#     detgA_run[i, :] = cumsum(detgAs) ./ collect(1:sweeps)
# end
# push!(S2_conv, vec(-log.(prod(detgA_run, dims=1))))
# end

# jldopen("$(path_dst)" * "EtgEnt_nosgn_LA2_U2.0_beta50.0.jld", "w") do file
#     # write(file, "filling", filling_list)
#     write(file, "S2_avg", S2_avg)
#     write(file, "S2_err", S2_err)
#     write(file, "S2_conv", S2_conv)
# end