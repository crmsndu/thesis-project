module hamming

using LinearAlgebra

function ham_weight(word)
    l = length(word)
    ham_weight = 0
    for i = 1:l
        if (word[i] == 1)
            ham_weight += 1
        end
    end
    return ham_weight
end

function v2i(x)
    l = length(x)
    tot = 0
    for i = 1:l
        tot += x[i] * 2^(i - 1)
    end
    return tot
end

function create_GH(order)
    n = 2^order
    k = n - order - 1
    P = Array{Int64}(undef, 0, n - k)
    for i = 1:n
        v = digits(i, base=2, pad=n - k - 1)
        if (ham_weight(v) < 2)
            continue
        end
        reverse!(v)
        v = hcat(v', iseven(ham_weight(v)))
        P = vcat(P, v)
    end
    G = hcat(Matrix{Int64}(I, k, k), P)
    H = hcat(P', Matrix{Int64}(I, n - k, n - k))
    return G, H
end

function get_h(H)
    n = size(H)[2]
    k = n - size(H)[1]
    h = Vector{Int64}(undef, n)
    h_inv = zeros(Int64, 2^(n - k))
    for i = 1:n
        h[i] = v2i(H[:, i])
        h_inv[v2i(H[:, i])] = i
    end
    return h, h_inv
end

end