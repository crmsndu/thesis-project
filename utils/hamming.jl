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

end