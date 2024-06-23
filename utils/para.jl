using Distributed


@everywhere module myfunc
f(x) = x^3 + randn()
end
@everywhere begin
    include("MyFunc.jl")
    import .myfunc, .myFunctions
    k = 3
    function f(x)
        return x + k + randn()
    end
end

# @time result = myfunc.f(3)
# println(result)

function f_para01b(n)
    res = @distributed (+) for i = 1:n
        f(3) + myfunc.f(3) + myFunctions.f(3) + k
    end
    res /= n
    return res
end

@time f_para01b(1_000)
@time result = f_para01b(100_000_000)

println(result)