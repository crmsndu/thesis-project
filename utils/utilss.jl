module utilss

export v2i

function v2i(x)
    l = length(x)
    tot = 0
    for i = 1:l
        tot += x[i] * 2^(i - 1)
    end
    return tot
end

end