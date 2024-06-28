using LinearAlgebra, Plots, DelimitedFiles, Distributed
@everywhere include("hamming.jl")
@everywhere include("chase.jl")
@everywhere import .hamming, .chase


@everywhere begin
    const G, H = hamming.create_GH(6)
    const k, n = size(G)
    const batch_size = 100
    # min_num = 30
    const iter_num = 4
    const window_size = 7
    const Α = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0, 1.0]
    const h = chase.compute_h(H)
    const h_inv = chase.compute_h_inv(H)
    const p = 3
end

@everywhere function turbo(Y)
    W = zeros(Float64, n, n)
    R = Y + Α[1] * W
    for i = 1:iter_num
        # R = R + Α[i] * W
        if (isodd(i))
            for j = 1:n
                W[j, :] = chase.SISO_Pyndiah_yes_h(n, h, h_inv, R[j, :], i, p) - Y[j, :]
            end
        end
        if (iseven(i))
            for j = 1:n
                W[:, j] = chase.SISO_Pyndiah_yes_h(n, h, h_inv, R[:, j], i, p) - Y[:, j]
            end
        end
        R = Y + Α[i+1] * W
    end
    D = ifelse.(R .> 0, 0, 1)
    return D
end

@everywhere function one_iter(σ, algorithm)
    m = rand((0, 1), k, k)
    x = mod.(G' * m, 2)
    x = mod.(x * G, 2)
    y = (-1) .^ x + randn(size(x)) * σ
    z = algorithm(y)
    return sum(mod.(x + z, 2))
end

@everywhere function error_rate_product(σ, algorithm)
    tmp = 0
    for i = 1:batch_size
        ttt = one_iter(σ, algorithm)
        if (ttt == 0)
            continue
        end
        tmp += ttt
        if mod(i, 10) == 0
            GC.gc()
        end
    end
    p = tmp / (batch_size * n * n)
    return p
end

function f_para01b(nn)
    res = @distributed (+) for i = 1:nn
        # GC.gc()
        error_rate_product(10^-0.35, turbo)
    end
    res /= nn
    return res
end

@time f_para01b(1)
@time result = f_para01b(3200)

println(result)