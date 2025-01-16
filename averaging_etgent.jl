
using JLD, Measurements, Statistics

function cumavg(data)
    return cumsum(data) ./ collect(1:length(data))
end

path_src = "./data_with_sgn/3x3/etgent_withreset/"
path_dst = "./data_with_sgn/3x3/processed/"

filling_list = [(f, f) for f in 1:8]
sgnprob_conv = Vector{Float64}[]
S2_signed_conv = Vector{Float64}[]
S2_unsigned_conv = Vector{Float64}[]

# merge data
filelist = [filter(x -> match(Regex("EtgEnt_LA3_Nup$(filling[1])_Ndn$(filling[2])_U2.0_beta6.0_*"), x) !== nothing, readdir(path_src)) for filling in filling_list]
if isempty(filelist[1])
    throw("no files found")
end
# detgA_list = Vector{Float64}[]
# numer_list = Vector{Float64}[]
# denom_list = Vector{Float64}[]

# for each lambda
for filenames in filelist
    data = load.("$(path_src)" .* filenames)

    # merge data across seeds
    sgnprob_data = vcat([data[i]["sgnprob"] for i in 1:length(filenames)]...)
    detgA_data = vcat([data[i]["sgndetgA"] .* data[i]["absdetgA"] for i in 1:length(filenames)]...)

    push!(sgnprob_conv, cumavg(sgnprob_data))

    push!(S2_unsigned_conv, -log.(abs.(cumavg(detgA_data))))

    push!(S2_signed_conv, -log.(abs.(cumavg(detgA_data .* sgnprob_data) ./ replace(cumavg(sgnprob_data), 0 => 1))))


    # push!(detgA_list, detgA_data)
    # push!(numer_list, detgA_data .* sgnprob_data)
    # push!(denom_list, sgnprob_data)
end

# stats
# detgA_avg = mean.(detgA_list)
# detgA_err = std.(detgA_list) ./ length(detgA_list)
# detgA = measurement.(detgA_avg, detgA_err)

# numer_avg = mean.(numer_list)
# numer_err = std.(numer_list) ./ length(numer_list)
# numer = measurement.(numer_avg, numer_err)

# denom_avg = mean.(denom_list)
# denom_err = std.(denom_list) ./ length(denom_list)
# denom = measurement.(denom_avg, denom_err)

# S2_unsigned = -log(prod(detgA))
# S2_signed = -log(prod(numer ./ denom))

# @show S2_unsigned.err

# detgA_avg = mean.(detgA_list)
# detgA_err = std.(detgA_list) ./ length(detgA_list)
# expmS2 = measurement.(detgA_avg, detgA_err)
# S2 = -log(prod(expmS2))

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

jldopen("$(path_dst)" * "EtgEnt_WithReset_LA3_U2.0_beta6.0.jld", "w") do file
    write(file, "filling", filling_list)
    write(file, "sgnprob_conv", sgnprob_conv)
    write(file, "S2_signed_conv", S2_signed_conv)
    write(file, "S2_unsigned_conv", S2_unsigned_conv)
end

