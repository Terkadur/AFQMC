
using JLD, Measurements, Statistics

path_src = "./data_with_sgn/2x2/etgent_Nk1/"
path_dst = "./data_with_sgn/2x2/processed/"
lambda_list = [0.0] #collect(0.0:0.2:0.8)
lambdas = length(lambda_list)

S2_avg = zeros(Float64, 1)
S2_err = zeros(Float64, 1)

S2_conv = Vector{Float64}[]

# merge data
filelist = [filter(x -> match(Regex("EtgEnt_LA2_Nup1_Ndn2_U2.0_lambda$(lambda)_beta18.0_*"), x) !== nothing, readdir(path_src)) for lambda in lambda_list]
if isempty(filelist[1])
    throw("no files found")
end
detgA_list = Vector{Float64}[]
numer_list = Vector{Float64}[]
denom_list = Vector{Float64}[]

# for each lambda
for filenames in filelist
    data = load.("$(path_src)" .* filenames)

    # merge data across seeds
    sgnprob_data = vcat([data[i]["sgnprob"] for i in 1:length(filenames)]...)
    detgA_data = vcat([data[i]["sgndetgA"] .* data[i]["absdetgA"] for i in 1:length(filenames)]...)

    push!(detgA_list, detgA_data)
    push!(numer_list, detgA_data .* sgnprob_data)
    push!(denom_list, sgnprob_data)
end

# stats
detgA_avg = mean.(detgA_list)
detgA_err = std.(detgA_list) ./ length(detgA_list)
detgA = measurement.(detgA_avg, detgA_err)

numer_avg = mean.(numer_list)
numer_err = std.(numer_list) ./ length(numer_list)
numer = measurement.(numer_avg, numer_err)

denom_avg = mean.(denom_list)
denom_err = std.(denom_list) ./ length(denom_list)
denom = measurement.(denom_avg, denom_err)

S2_unsigned = -log(prod(detgA))
S2_signed = -log(prod(numer ./ denom))

# detgA_avg = mean.(detgA_list)
# detgA_err = std.(detgA_list) ./ length(detgA_list)
# expmS2 = measurement.(detgA_avg, detgA_err)
# S2 = -log(prod(expmS2))

@show S2_unsigned.val
@show S2_unsigned.err

@show S2_signed.val
@show S2_signed.err

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

# sweeps = length(numer_list[1])
# @assert length(denom_list[1]) == sweeps "Unequal number of sweeps for numer and denom"

# numer_run = cumsum(numer_list[1]) ./ collect(1:sweeps)
# denom_run = cumsum(denom_list[1]) ./ collect(1:sweeps)
# push!(S2_conv, vec(-log.(abs.(numer_run ./ denom_run))))

# jldopen("$(path_dst)" * "EtgEnt_Nup1_Ndn1_LA2_U2.0_beta10.0_Nk1.jld", "w") do file
#     # write(file, "filling", filling_list)
#     write(file, "S2_conv", S2_conv)
# end

