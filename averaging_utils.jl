
using JLD, Measurements, LinearAlgebra, Statistics
"""
    sum_anti_diag!(v, A)

    Sum the matrix elements over anti-diagonal directions, including
all super/sub ones
"""
function sum_anti_diag!(v::AbstractVector, A::AbstractMatrix)
    row, col = size(A)
    row == col || @error "Non-sqaure matrix"

    for (idx, i) = enumerate(-col+1:col-1)
        if i < 0
            v[idx] = sum([A[j, col+1+i-j] for j = 1:col+i])
        elseif i > 0
            v[idx] = sum([A[j, col+1+i-j] for j = 1+i:col])
        else
            v[idx] = sum([A[j, col+1+i-j] for j = 1:col])
        end
    end

    return v
end

"""
    sum_diag!(v, A)

    Sum the matrix elements over diagonal directions, including
all super/sub ones
"""
function sum_diag!(v::AbstractVector, A::AbstractMatrix)
    row, col = size(A)
    row == col || @error "Non-sqaure matrix"

    for (idx, i) = enumerate(-col+1:col-1)
        if i < 0
            v[idx] = sum([A[j, j+i] for j = -i+1:col])
        elseif i > 0
            v[idx] = sum([A[j-i, j] for j = i+1:col])
        else
            v[idx] = sum([A[j, j] for j = 1:col])
        end
    end

    return v
end

function get_probs_raw(data::Dict)
    Pn2_up = data["Pn2_up"]
    Pn2_dn = data["Pn2_dn"]

    values = size(Pn2_up, 1)
    samples = size(Pn2_up, 2)

    Pn2_raw = zeros(ComplexF64, 2 * values - 1, samples)
    Pm2_raw = zeros(ComplexF64, 2 * values - 1, samples)
    Pmn2_raw = zeros(ComplexF64, values, values, samples)

    for i in axes(Pn2_up, 2)
        @views kron!(Pmn2_raw[:, :, i], reshape(Pn2_up[:, i], values, 1), reshape(Pn2_dn[:, i], 1, values))
        @views sum_anti_diag!(Pn2_raw[:, i], Pmn2_raw[:, :, i])
        @views sum_diag!(Pm2_raw[:, i], Pmn2_raw[:, :, i])
    end

    return Pmn2_raw, reverse(Pn2_raw; dims=1), Pm2_raw
end

function get_probs_stats(Pmn2_raw::AbstractArray, Pn2_raw::AbstractArray, Pm2_raw::AbstractArray)
    Pmn2_avg = mean(Pmn2_raw, dims=3)
    Pmn2_err = std(Pmn2_raw, dims=3) / sqrt(size(Pmn2_raw, 3))

    Pn2_avg = mean(Pn2_raw, dims=2)
    Pn2_err = std(Pn2_raw, dims=2) / sqrt(size(Pn2_raw, 2))

    Pm2_avg = mean(Pm2_raw, dims=2)
    Pm2_err = std(Pm2_raw, dims=2) / sqrt(size(Pm2_raw, 2))

    return real.(Pmn2_avg), real.(Pmn2_err), real.(Pn2_avg), real.(Pn2_err), real.(Pm2_avg), real.(Pm2_err)
end

function JackknifeObservable(x::AbstractVector{T}; J=zeros(T, length(x))) where {T}
    length(x) > 1 || throw(ArgumentError("The sample must have size > 1"))

    x_sum = sum(x)
    l = length(x)

    for i in eachindex(x)
        J[i] = (x_sum - x[i]) / (l - 1)
    end

    return J
end

Hα(α::E, Pnα::AbstractArray{T}) where {T,E} = begin
    # regular Shannon entropy -∑p⋅lnp
    α == 1 && return -sum(Pnα .* log.(Pnα))
    α == 1.0 && return -sum(Pnα .* log.(Pnα))

    return log(sum(Pnα .^ α)) / (1 - α)
end

function estimate_Hα(Pn::AbstractArray{T}; α::Float64=0.5) where {T}

    Pn_list = zeros(T, size(Pn))
    Hα_jack = zeros(T, length(axes(Pn, 2)))
    for i in axes(Pn, 1)
        Pn_list[i, :] = JackknifeObservable(@view Pn[i, :])
    end

    for i in axes(Pn, 2)
        Hα_jack[i] = Hα(α, @view Pn_list[:, i])
    end

    Hα_avg = mean(Hα_jack)
    Hα_err = std(Hα_jack) * sqrt(length(Hα_jack))

    return Hα_avg, Hα_err
end

function get_shannon_stats(Pn2::AbstractArray, Pm2::AbstractArray)
    Hn_avg, Hn_err = estimate_Hα(Pn2)
    Hm_avg, Hm_err = estimate_Hα(Pm2)

    return real(Hn_avg), real(Hn_err), real(Hm_avg), real(Hm_err)
end