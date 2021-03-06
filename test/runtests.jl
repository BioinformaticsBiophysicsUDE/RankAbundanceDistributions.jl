using Base
using Test 
using Distributed
using CSV
using Statistics
using DataFrames
using RankAbundanceDistributions

# write your own tests here

procIDs = addprocs(2)
@everywhere using DataFrames
@everywhere using RankAbundanceDistributions

test_RAD_1 = DataFrame(sample = fill("test", 1000),
                       rank = collect(1:1000),
                       abundance = fill(1, 1000));
test_RAD_2 = DataFrame(sample = fill("test", 100),
                       rank = collect(1:100),
                       abundance = fill(0.01, 100),
                       lower_ci = fill(0.01, 100),
                       upper_ci = fill(0.01, 100));

#test reading OTU table and transforming it to RAD:
c1 = CSV.read("part_c_patient_1.csv",header=1);
c2 = CSV.read("part_c_patient_2.csv",header=1);
rads_c1 = RAD_set_from_OTU_table(c1,1);
rads_c2 = RAD_set_from_OTU_table(c2,1);
rads = vcat(rads_c1,rads_c2);
rads = rads[rads[:sample].!= "IgMonly_CD27_D_kaks",:];
rads[:sample] = convert(Array{String}, rads[:sample]);

#should find 16 samples in set, with 604 ranks in total:
samples = unique(rads[:sample]);
@test length(samples) == 16
@test length(rads[:sample]) == 604

#we should obtain the same result as with a direct reading of the original RADs:
orig_rads = read_RAD_set("orig-RADs.csv.gz");
@test orig_rads == rads

NRADorig = read_RAD_set("NRADs-B-cells.csv");
@test size(NRADorig) == (560,5)
@test typeof(NRADorig[:sample]) == Array{String,1}

@test test_RAD_1 == produce_uniform_RAD("test", 1000, 1)

#test_RAD_2 is the outcome of the MaxRank normalization of test_RAD_1:
@test test_approx_abundance_equality(test_RAD_2,
                                     RAD_set_normalize(test_RAD_1, n=100, R=100))

#test_RAD_1 is *not* the outcome of its own MaxRank normalization (normalized RAD has fewer ranks):
@test !test_approx_abundance_equality(test_RAD_1,
                                      RAD_set_normalize(test_RAD_1, n=100, R=100))

# test whether normalized RADs have abundance summing up to 1.0:
@test isapprox(1.0, sum(RAD_set_normalize(produce_random_RAD("rnd",100,1000), n=2)[:abundance]))

# test results for a simple RAD for which we know the theoretical result:
S = 3; #original richness
A = 2; #original abundance
R = 2; #MaxRank
n = 100000; #averaging over n NRADs
RAD = produce_uniform_RAD("uni", S, A);
NRAD = RAD_set_normalize(RAD,R=R,n=n);
#these are statistical tests that may fail in rare cases:
@test 0.566 < NRAD[1,:abundance] < 0.567
@test 0.433 < NRAD[2,:abundance] < 0.434

#test MaxRank normalization:
NRADnew = RAD_set_normalize(rads, n=200, verbose=true);

for s in samples
    diffs = NRADorig[NRADorig[:sample].==s,:abundance].-
    NRADnew[NRADnew[:sample].==s,:abundance]
    #these are statistical tests that may fail in rare cases:
    @test abs(mean(diffs))<1.0e-10
    @test std(diffs)<1.0e-3
end

#test parallelized MaxRank normalization
println("\ntesting parallelized algorithm")
NRADnew = RAD_set_normalize(rads, n=200, procIDs=procIDs, verbose=true);
for s in samples
    diffs = NRADorig[NRADorig[:sample].==s,:abundance].-
    NRADnew[NRADnew[:sample].==s,:abundance]
    #these are statistical tests that may fail in rare cases:
    @test abs(mean(diffs))<1.0e-10
    @test std(diffs)<1.0e-3
end

println("\n --- successfully completed tests of MaxRank algorithms ---\n")

@test size(RAD_samples_subset(rads,["IgG_A";"IgG_B"]))[1]==74

#split RAD set into files, read, and reassemble:
split_RAD_set_into_files(rads, "myRADs", 3)
d1 = read_RAD_set("myRADs-1.csv.gz");
d2 = read_RAD_set("myRADs-2.csv.gz");
d3 = read_RAD_set("myRADs-3.csv.gz");
#compare reassembled RAD with original:
@test rads == vcat(d1,d2,d3)
