using Test
using ReferenceTests
using BSON

include("../scripts/diffusion_nl_1D.jl")

## Unit Tests
@testset "av Function tests" begin
    @test av([20,30,40]) ≈ [0.5*(20+30), 0.5*(30+40)]
    @test av([20,30,40]) == [0.5*(20+30), 0.5*(30+40)]
    
    @test av([1,2,3,4,5]) ≈ [0.5*(1+2), 0.5*(2+3), 0.5*(3+4), 0.5*(4+5)]
    @test av([1,2,3,4,5]) == [0.5*(1+2), 0.5*(2+3), 0.5*(3+4), 0.5*(4+5)]

    @test av([-1,1,-1,-2,-4]) ≈ [0.5*(-1+1), 0.5*(1 + (-1)), 0.5*(-1-2), 0.5*(-2-4)]
    @test av([-1,1,-1,-2,-4]) == [0.5*(-1+1), 0.5*(1 + (-1)), 0.5*(-1-2), 0.5*(-2-4)]
end

# Function to compare all dict entries
comp(d1,d2) = keys(d1) == keys(d2) &&
    all([v1 ≈ v2 for (v1,v2) in zip(values(d1), values(d2))])

# Run Model
H, qx = diffusion_1D()

indices_dict = BSON.load("./reftest-truths/indices.bson")
H_idx = indices_dict[":H"]
qx_idx = indices_dict[":qx"]

d_test = Dict(":H" => H[H_idx], ":qx" => qx[qx_idx])
@testset "Ref-tests" begin
    @test_reference "reftest-truths/truth.bson" d_test by=comp
end