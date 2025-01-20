
using JLD, Measurements, Statistics

function cumavg(data)
    return cumsum(data) ./ collect(1:length(data))
end

path_src = "./data_with_new_sgn/3x3/etgent/"
path_dst = "./data_with_new_sgn/3x3/processed/"

filling_list = [(4, 4), (5, 5)]
sgnprob_prod_conv = Vector{Float64}[]
sgnprob_switch_conv = Vector{Float64}[]
S2_prod_conv = Vector{Float64}[]
S2_switch_conv = Vector{Float64}[]
S2_unsigned_conv = Vector{Float64}[]

# merge data
filelist = [filter(x -> match(Regex("EtgEnt_LA3_Nup$(filling[1])_Ndn$(filling[2])_U2.0_beta6.0_*"), x) !== nothing, readdir(path_src)) for filling in filling_list]
if isempty(filelist[1])
    throw("no files found")
end

for filenames in filelist
    data = load.("$(path_src)" .* filenames)

    # merge data across seeds
    sgnprob1_data = vcat([data[i]["sgnprob1"] for i in 1:length(filenames)]...)
    sgnprob2_data = vcat([data[i]["sgnprob2"] for i in 1:length(filenames)]...)
    sgnprob_prod_data = sgnprob1_data .* sgnprob2_data
    sgnprob_switch_data = zeros(size(sgnprob_prod_data))
    for i in eachindex(sgnprob_switch_data)
        if (i-1) % 128 <= 127
            sgnprob_switch_data[i] = sgnprob1_data[i]
        else
            sgnprob_switch_data[i] = sgnprob2_data[i]
        end
    end

    detgA_data = vcat([data[i]["sgndetgA"] .* data[i]["absdetgA"] for i in 1:length(filenames)]...)

    sgnprob_prod_conv_list = cumavg(sgnprob_prod_data)
    push!(sgnprob_prod_conv, sgnprob_prod_conv_list)

    sgnprob_switch_conv_list = cumavg(sgnprob_switch_data)
    push!(sgnprob_switch_conv, sgnprob_switch_conv_list)


    S2_unsigned_conv_list = cumavg(detgA_data)
    for i in eachindex(S2_unsigned_conv_list)
        if S2_unsigned_conv_list[i] <= 0
            S2_unsigned_conv_list[i] = 0
        else
            S2_unsigned_conv_list[i] = -log(S2_unsigned_conv_list[i])
        end
    end
    push!(S2_unsigned_conv, S2_unsigned_conv_list[1:16:end])


    S2_prod_conv_list = cumavg(detgA_data .* sgnprob_prod_data)
    for i in eachindex(S2_prod_conv_list)
        if sgnprob_prod_conv_list[i] == 0
            S2_prod_conv_list[i] = 0
        else
            S2_prod_conv_list[i] /= sgnprob_prod_conv_list[i]
            if S2_prod_conv_list[i] <= 0
                S2_prod_conv_list[i] = 0
            else
                S2_prod_conv_list[i] = -log(S2_prod_conv_list[i])
            end
        end
    end
    push!(S2_prod_conv, S2_prod_conv_list[1:16:end])

    S2_switch_conv_list = cumavg(detgA_data .* sgnprob_switch_data)
    for i in eachindex(S2_switch_conv_list)
        if sgnprob_switch_conv_list[i] == 0
            S2_switch_conv_list[i] = 0
        else
            S2_switch_conv_list[i] /= sgnprob_switch_conv_list[i]
            if S2_switch_conv_list[i] <= 0
                S2_switch_conv_list[i] = 0
            else
                S2_switch_conv_list[i] = -log(S2_switch_conv_list[i])
            end
        end
    end
    push!(S2_switch_conv, S2_switch_conv_list[1:16:end])
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

jldopen("$(path_dst)" * "EtgEnt_LA3_U2.0_beta6.0.jld", "w") do file
    write(file, "filling", filling_list)
    write(file, "sgnprob_prod_conv", sgnprob_prod_conv)
    write(file, "sgnprob_switch_conv", sgnprob_switch_conv)
    write(file, "S2_prod_conv", S2_prod_conv)
    write(file, "S2_switch_conv", S2_switch_conv)
    write(file, "S2_unsigned_conv", S2_unsigned_conv)
end

