using Test, ReferenceTests, BSON

include("../scripts/viscous_NS_2D.jl")

## Reference Tests with ReferenceTests.jl
# We put both arrays xc and P into a BSON.jl and then compare them

"Compare all dict entries."
comp(d1,d2) = keys(d1) == keys(d2) && all([ v1 ≈ v2 for (v1,v2) in zip(values(d1), values(d2))])

X, P = acoustic_2D()

inds = Int.(ceil.(LinRange(1, length(X), 12)))
d = Dict(:X => X[inds], :P => P[inds])

@testset "Ref-tests"
@test_reference "reftest-files/X.bson" d by=comp
end