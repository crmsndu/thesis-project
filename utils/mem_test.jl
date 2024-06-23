struct Container
    field::Vector{Matrix{Float64}}
end
function f!(x::Container)
    push!(x.field, randn(10, 10000))
end
function g!(x::Container, n::Int64)
    for _ in 1:n
        f!(x)
    end
end

X = randn(10, 5000);
c = Container([X]);
g!(c, 5000)
GC.gc()
# c is about 7.5GiB in size now, and I have about 8.3GB in memory occupied.

c = nothing
GC.gc()