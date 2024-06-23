# include("utilss.jl")

# using .utilss

module chase

function v2i(x)
    l = length(x)
    tot = 0
    for i = 1:l
        tot += x[i] * 2^(i - 1)
    end
    return tot
end

function compute_h_inv(H)
    n = size(H)[2]
    k = n - size(H)[1]
    h_inv = zeros(Int64, 2^(n - k))
    for i = 1:n
        h_inv[v2i(H[:, i])] = i
    end
    return h_inv
end

function compute_h(H)
    n = size(H)[2]
    h = Vector{Int64}(undef, n)
    for i = 1:n
        h[i] = v2i(H[:, i])
    end
    return h
end

Β = [0.2, 0.4, 0.6, 0.8, 1.0, 1.0, 1.0, 1.0]

function SISO_Pyndiah(H, y, iter)
    n = size(H)[2]
    x = zeros(Int64, n)
    s = Int64(0)
    h = compute_h(H)
    h_inv = compute_h_inv(H)
    for i = 1:n
        if (y[i] < 0)
            x[i] = 1
            s = xor(s, h[i])
        end
    end
    α = abs.(y)
    max_weight = -9999
    best_match = copy(x)
    p = 4
    rnk = partialsortperm(α, (length(α)-p+1):length(α), rev=true)   # rnk is the array of the p least reliable positions
    weight = -9999 * ones(2^p)
    for i = 0:2^p-1
        xt = copy(x)
        st = copy(s)
        for j = 0:p-1
            if (i & (2^j) == (2^j))
                xt[rnk[j+1]] = 1 - xt[rnk[j+1]]
                st = xor(st, h[rnk[j+1]])
            end
        end
        if (st != 0)
            idx = h_inv[st]
            if (idx != 0)
                xt[idx] = 1 - xt[idx]
                weight[i+1] = sum(((-1) .^ xt) .* y)
                if (weight[i+1] > max_weight)
                    best_match = copy(xt)
                    max_weight = sum(((-1) .^ xt) .* y)
                end
            end
        else
            weight[i+1] = sum(((-1) .^ xt) .* y)
            if (weight[i+1] > max_weight)
                best_match = copy(xt)
                max_weight = sum(((-1) .^ xt) .* y)
            end
        end
    end
    maxw = -9999 * ones(n)
    for i = 0:2^p-1
        xt = copy(x)
        st = copy(s)
        for j = 0:p-1
            if (i & (2^j) == (2^j))
                xt[rnk[j+1]] = 1 - xt[rnk[j+1]]
                st = xor(st, h[rnk[j+1]])
            end
        end
        if (st != 0)
            idx = h_inv[st]
            if (idx != 0)
                xt[idx] = 1 - xt[idx]
                for j = 1:p
                    if (xt[rnk[j]] != best_match[rnk[j]] && weight[i+1] > maxw[rnk[j]])
                        maxw[rnk[j]] = weight[i+1]
                    end
                end
                if (xt[idx] != best_match[idx] && weight[i+1] > maxw[idx])
                    maxw[idx] = weight[i+1]
                end
            end
        else
            for j = 1:p
                if (xt[rnk[j]] != best_match[rnk[j]] && weight[i+1] > maxw[rnk[j]])
                    maxw[rnk[j]] = weight[i+1]
                end
            end
        end
    end
    r = zeros(Float64, n)
    for i = 1:n
        if (maxw[i] != -9999)
            r[i] = (max_weight - maxw[i]) / 2
        else
            r[i] = Β[iter]
        end
    end
    ans = ((-1) .^ best_match) .* r
    return ans
end

function SIHO_Pyndiah(H, y, iter)
    n = size(H)[2]
    x = zeros(Int64, n)
    s = Int64(0)
    h = compute_h(H)
    h_inv = compute_h_inv(H)
    for i = 1:n
        if (y[i] < 0)
            x[i] = 1
            s = xor(s, h[i])
        end
    end
    α = abs.(y)
    max_weight = -9999
    best_match = copy(x)
    p = 0
    rnk = partialsortperm(α, (length(α)-p+1):length(α), rev=true)   # rnk is the array of the p least reliable positions
    weight = -9999 * ones(2^p)
    for i = 0:2^p-1
        xt = copy(x)
        st = copy(s)
        for j = 0:p-1
            if (i & (2^j) == (2^j))
                xt[rnk[j+1]] = 1 - xt[rnk[j+1]]
                st = xor(st, h[rnk[j+1]])
            end
        end
        if (st != 0)
            idx = h_inv[st]
            if (idx != 0)
                xt[idx] = 1 - xt[idx]
                weight[i+1] = sum(((-1) .^ xt) .* y)
                if (weight[i+1] > max_weight)
                    best_match = copy(xt)
                    max_weight = sum(((-1) .^ xt) .* y)
                end
            end
        else
            weight[i+1] = sum(((-1) .^ xt) .* y)
            if (weight[i+1] > max_weight)
                best_match = copy(xt)
                max_weight = sum(((-1) .^ xt) .* y)
            end
        end
    end
    maxw = -9999 * ones(n)
    for i = 0:2^p-1
        xt = copy(x)
        st = copy(s)
        for j = 0:p-1
            if (i & (2^j) == (2^j))
                xt[rnk[j+1]] = 1 - xt[rnk[j+1]]
                st = xor(st, h[rnk[j+1]])
            end
        end
        if (st != 0)
            idx = h_inv[st]
            if (idx != 0)
                xt[idx] = 1 - xt[idx]
                for j = 1:p
                    if (xt[rnk[j]] != best_match[rnk[j]] && weight[i+1] > maxw[rnk[j]])
                        maxw[rnk[j]] = weight[i+1]
                    end
                end
                if (xt[idx] != best_match[idx] && weight[i+1] > maxw[idx])
                    maxw[idx] = weight[i+1]
                end
            end
        else
            for j = 1:p
                if (xt[rnk[j]] != best_match[rnk[j]] && weight[i+1] > maxw[rnk[j]])
                    maxw[rnk[j]] = weight[i+1]
                end
            end
        end
    end
    r = zeros(Float64, n)
    for i = 1:n
        if (maxw[i] != -9999)
            r[i] = (max_weight - maxw[i]) / 2
        else
            r[i] = Β[iter]
        end
    end
    ans = ((-1) .^ best_match) .* r
    return ans
end

end