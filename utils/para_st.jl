using LinearAlgebra, Plots, DelimitedFiles, Distributed
@everywhere include("hamming.jl")
@everywhere include("chase.jl")
@everywhere import .hamming, .chase


@everywhere begin
    const G, H = hamming.create_GH(9)
    const k, n = size(G)
    const batch_size = 100
    min_num = 30
    iter_num = 6
    window_size = 7
    const Α = [0.0, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.0]
    const h = chase.compute_h(H)
    const h_inv = chase.compute_h_inv(H)
end

@everywhere function error_rate(σ::Float64)
    nn = 4
    res = 0.0
    for ii = 1:nn
        if mod(ii, 2) == 0
            GC.gc()
        end
        ttt = 0
        n2 = Int64(n / 2)
        A = zeros(Int64, batch_size + 12, n2, n2)
        for i = 2:size(A)[1]
            m = rand((0, 1), n2, k - n2)
            p = mod.([A[i-1, :, :]' m] * G[:, k+1:n], 2)
            A[i, :, :] = [m p]
        end
        B = (-1.0) .^ A
        B = B + randn(size(B)) * σ
        R = copy(B)
        a = zeros(size(B))
        for i = 7:size(B)[1]
            for _ = 1:1
                for j = i:-1:i-5
                    for l = 1:n2
                        w = zeros(Float64, n)
                        w = chase.SISO_Pyndiah_yes_h(n, h, h_inv, vec([R[j-1, :, l] R[j, l, :]]), i - j + 1) - vec([B[j-1, :, l] B[j, l, :]])
                        R[j-1, :, l] = B[j-1, :, l] + w[1:n2] * Α[i-j+2]
                        R[j, l, :] = B[j, l, :] + w[n2+1:n] * Α[i-j+2]
                        w = nothing
                    end
                end
            end
        end
        D = ifelse.(R .> 0, 0, 1)
        for i = 7:size(A)[1]-6
            if (D[i, :, :] != A[i, :, :])
                ttt += sum(mod.(D[i, :, :] + A[i, :, :], 2))
            end
        end
        p = ttt / (batch_size * n / 2 * n / 2)
        res += p
    end
    return res / nn
end

function f_para01b(nn)
    res = @distributed (+) for i = 1:nn
        # GC.gc()
        error_rate(10^-0.4)
    end
    res /= nn
    return res
end

@time f_para01b(1)
@time result = f_para01b(16)

println(result)