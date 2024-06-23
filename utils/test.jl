using Distributed

# 添加工作进程
# addprocs(4)  # 添加4个工作进程，你可以根据需要调整这个数字

@everywhere function myfunc()::Float64
    # 你的函数定义
    return randn() * 0.1  # 示例函数，返回一个随机数
end

function f_para01b(n::Int64)
    res::Float64 = @distributed (+) for _ = 1:n
        myfunc()
    end
    res /= n
    return res
end
@time f_para01b(1_000)
@time f_para01b(1_000_000_000)