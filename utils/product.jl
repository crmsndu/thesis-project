using LinearAlgebra, Plots, DelimitedFiles, Distributed
include("hamming.jl")
include("chase.jl")
import .hamming, .chase

const G, H = hamming.create_GH(6)
const k, n = size(G)
const batch_size = 100
const min_num = 30
const iter_num = 8
const window_size = 7
const Α = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0]
const h = chase.compute_h(H)
const h_inv = chase.compute_h_inv(H)

function turbo(Y)
    W = zeros(Float64, n, n)
    R = Y + Α[1] * W
    for i = 1:iter_num
        # R = R + Α[i] * W
        if (isodd(i))
            for j = 1:n
                W[j, :] = chase.SISO_Pyndiah_yes_h(n, h, h_inv, R[j, :], i) - Y[j, :]
            end
        end
        if (iseven(i))
            for j = 1:n
                W[:, j] = chase.SISO_Pyndiah_yes_h(n, h, h_inv, R[:, j], i) - Y[:, j]
            end
        end
        R = Y + Α[i+1] * W
    end
    D = ifelse.(R .> 0, 0, 1)
    return D
end

function one_iter(σ, algorithm)
    m = rand((0, 1), k, k)
    x = mod.(G' * m, 2)
    x = mod.(x * G, 2)
    y = (-1) .^ x + randn(size(x)) * σ
    z = algorithm(y)
    return sum(mod.(x + z, 2))

end
one_iter(1, turbo)

function error_rate_product(σ, algorithm)
    cnt = 0
    tmp = 0
    batch_num = 0
    while (cnt < min_num)
        tot = 0
        for i = 1:batch_size
            ttt = one_iter(σ, algorithm)
            if (ttt == 0)
                continue
            end
            tot += 1
            tmp += ttt
        end
        cnt += tot
        batch_num += 1
    end
    p = tmp / (batch_num * batch_size * n * n)
    return p
end

error_rate_product(10^-0.275, turbo)

result = error_rate_product(10^-0.275, turbo)

println(result)