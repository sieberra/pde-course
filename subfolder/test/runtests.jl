using Test, ReferenceTests, Random

@test 1 == 1

include("../scripts/diffusion_nl_1D.jl")

@testset "av-test" begin
    @test av( [1 2 3 3]) == [1.5, 2.5, 3.0]
    @test av([ 10 10])  == [10]
    @test av([-1 1 2 -2]) == [0, 1.5, 0]
end

## Reference tests for H, qx

# fix the random number generator, so the indices are the same in every test.
Random.seed!(11);
index_H = Int64.(round.(rand(20).*(size(H).-1))).+1;  # -1, +1 makes sure the index is not 0, although the chance is small

# Repeat for qx
Random.seed!(12);
index_qx = Int64.(round.(rand(20).*(size(qx).-1))).+1;

# I find the test arranged in testsets to be more pleasing to look at in the output.
#@testset "random-indices-H-qx" begin
#    @test_reference "ref-tests/H2.test" H[index_H]       # save the values found for the indices in a test file
#    @test_reference "ref-tests/qx2.test" qx[index_qx]
#end

# repeat the reference test for arbitrary but not random indices
#@testset "arbitrary-indices" begin
#    @test_reference "ref-tests/H.test" H[[1,2,3,5,8,13,21,34,55,89,11,22,33,44,66,77,88,99,111,122]]
#    @test_reference "ref-tests/qx.test" qx[[1,2,3,5,8,13,21,34,55,89,11,22,33,44,66,77,88,99,111,122]]
#end