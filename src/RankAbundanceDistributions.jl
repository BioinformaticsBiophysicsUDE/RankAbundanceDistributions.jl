module RankAbundanceDistributions

# package code goes here
export brokenstick_RAD,
log10binmeans,
produce_random_RAD,
produce_uniform_RAD,
RAD_cmds_set_means,
RAD_downsample,
RAD_set_from_OTU_table,
RAD_leave_one_out_regression_forest,
RAD_log10log10coarser,
RAD_kmeans_cMDS,
RAD_kmedoids,
RAD_samples_subset,
RAD_samples_regex_set,
RAD_set_abundance_normalization,
RAD_set_abundance_sum,
RAD_set_affinityprop,
RAD_set_cMDS,
RAD_set_entropies,
RAD_set_evenness,
RAD_set_mean,
RAD_set_minimum_richness,
RAD_set_normalize,
RAD_set_plot,
RAD_set_representatives,
RAD_set_richness,
RAD_set_summary,
RAD_set_to_distmat,
RADs_average,
RADs_median,
RAD_subsets_for_procs,
RAD_set_from_matrix,
RAD_to_population_vector,
random_brokenstick_RAD,
random_powerlaw_RAD,
read_RAD_set,
split_RAD_set_into_files,
test_approx_abundance_equality

import Clustering
import CSV
import DataFrames
import DecisionTree
import Distances
import Distributed
import Distributions
import GZip
import LinearAlgebra
import MultivariateStats
import Plots
import Random
import Statistics
import StatPlots
import StatsBase

"""
Broken stick species abundance distribution

Input:

- richness S

- number of individuals N

- optional: should distribution be normalized to total probability 1 (default = false)

Output:

- abundance vector of length S telling how frequent species with a certain abundance n are
"""

function brokenstick_species_abundance(
                                       S::Int64,
                                       N::Int64,
                                       normalize::Bool=false
                                       )

    if S <= 0 || N <= 0
        error("richness S and number of individuals N must both be positive integers")
    end
    
    a = map(n->(S*(S-1)/N)*(1-n/N)^(S-2), 1:S) #abundance
    if normalize == true #normalize to total probability = 1
        a = a ./ sum(a)
    end
    return a
end

"""
Produces a random abundance RAD data frame with given richness by drawing random integer numbers (uniformly distributed between 0 and max_abundance) and ranking them according to abundance.

Input:

- label: AbstractString to be put into sample-column of RAD data frame

- richness: number of ranks

- max_abundance: maximum abundance

Output:

- data frame of the random RAD

"""
function produce_random_RAD(label::String, richness::Int64, max_abundance::Int64)
    abundances = sort!(rand(1:max_abundance, richness), rev=true)
    return DataFrames.DataFrame(sample = fill(label, richness),
                     rank = collect(1:richness),
                     abundance = abundances)
end

"""
Produces a uniform abundance RAD data frame with constant "abundance" over "richness" ranks.

Input:

- label: String to be put into sample-column of RAD data frame

- richness: number of ranks

- abundance: value of constant abundance

Output:

- data frame of the uniform RAD
"""
function produce_uniform_RAD(label::String, richness::Int64, abundance::Int64)
    return DataFrames.DataFrame(sample = fill(label, richness),
                     rank = collect(1:richness),
                     abundance = fill(abundance, richness))
end

"""
MaxRank normalization of Rank Abundance Distributions (RADs)

Input:

- dataframe df with (at least) columns rank and abundance

- maximum rank R to which we will normalize

- number of random drawing rounds over which we will average the RAD

- conf_level: quantile size of confidence interval, e.g. 0.9 means CI between 0.05 and 0.95 quantile; if conf_level=0. (default), no CI is computed

- n_boots: number of bootstraps for the computation of the CI (default: 100)

Output:

- dataframe with columns

  - rank,

  - abundance (normalized),

  - lower_ci (lower boundary of confidence interval, or abundance (if CI=0.))

  - upper_ci (upper boundary of confidence interval, or abundance (if CI=0.))

Note: We will rarely call this function directly, but usually from function RAD_set_normalize.
"""

function MaxRank_normalize(
                           df::DataFrames.SubDataFrame, #dataframe containing at least columns "rank" and "abundance"
                           R::Int64, #max. rank to which we normalize
                           n::Int64, #runs to be averaged over
                           conf_level::Float64=0., #quantile size of bootstrapped confidence interval of abundance average, e.g. 0.9 is from 0.05 to 0.95 quantile
                           n_boots::Int64=100,
                           verbose::Bool=true #print progress information
    )
    if R > maximum(df[:rank])
        error("available ranks < requested maximum rank")
    end
    rmax = maximum(df[:rank])

    if verbose
        print(".")
    end
    
    if rmax == R #this is a case that is sometimes requested

        #In this case we have almost nothing to compute, except for the normalization to sum 1

        normalized_abundance = copy(df[:abundance])
        normalized_abundance = normalized_abundance / sum(normalized_abundance)

        return DataFrames.DataFrame(rank = collect(1:R), 
                         abundance = normalized_abundance, 
                         lower_ci = normalized_abundance, 
                         upper_ci = normalized_abundance) 
    end

    #Now the usual case with rmax > R:

    normalized_abundance = zeros(R, n)

    #roll out RAD into one population vector v:
    N_indivs = sum(df[:abundance])
    v = Array{Int64}(undef,N_indivs)
    N_start = 1
    for r in 1:rmax
        N_stop = N_start+df[r,:abundance]-1
        v[N_start:N_stop] .= r
        N_start = N_stop+1
    end

    for iter in 1:n #average over n RADs

        #shuffle v randomly:
        Random.shuffle!(v)

        vpos = (N_indivs+1.0)/2.
        vtryPos = 0
        Dvpos = vpos/2.
        Rtry = 0
        unique_elems = [1]
        
        #find position in v so that elements 1...Rtry have about richness R:
        
        while Rtry != R
            vtryPos = round(Int64,vpos)
            unique_elems = unique(v[1:vtryPos])
            Rtry = length(unique_elems)
            if Rtry < R
                vpos += Dvpos
            elseif Rtry > R
                vpos -= Dvpos
            end
            Dvpos = Dvpos/2.
        end
        
        vtryPos += 1
        next_elem = v[vtryPos]
        
        while in(next_elem, unique_elems)
            vtryPos += 1
            next_elem = v[vtryPos]
        end
        vtryPos -= 1
        
        dict = StatsBase.countmap(v[1:vtryPos])

        normalized_abundance[:,iter] = sort(collect(values(dict)),rev=true)
        
        #normalize to sum(p)=1
        normalized_abundance[:,iter] ./= sum(normalized_abundance[:,iter])
    end
    
    # average normalized abundance
    res_abund = vec(StatsBase.mean(normalized_abundance,dims=2))
    ci_lower = zeros(R)
    ci_upper = zeros(R)

    # bootstrap confidence intervals
    if conf_level > 0.
        quantile_lower = 0.5*(1.0 - conf_level)
        quantile_upper = 1.0 - quantile_lower
        means = zeros(n_boots)
        for j in 1:R
            #for each rank we bootstrap the means
            for i in 1:n_boots
                means[i] = StatsBase.mean(StatsBase.sample(normalized_abundance[j,:], n))
            end
            #... and compute the lower and upper quantile
            ci_lower[j] = StatsBase.quantile(means, quantile_lower)
            ci_upper[j] = StatsBase.quantile(means, quantile_upper)
        end
        return DataFrames.DataFrame(rank = 1:R, #ranks
                         abundance = res_abund, #mean normalized abundances
                         lower_ci = ci_lower, #lower confidence interval of mean
                         upper_ci = ci_upper)  #upper confidence interval of mean
    else #no confidence interval => copy res_abund to lower_ci and upper_ci
        return DataFrames.DataFrame(rank = 1:R, #ranks
                         abundance = res_abund, #mean normalized abundances
                         lower_ci = res_abund, 
                         upper_ci = res_abund) 
    end
       
end

"""
For a given RAD, produce a downsampled RAD with smaller richness R.

Input:

- DataFrame with columns rank and abundance

- richness R to which we downsample

- name that should appear in "sample" column of downsampled RAD (default: "downsampled")

Output:

- DataFrame with columns sample, rank, abundance

"""
function RAD_downsample(
                        df::DataFrames.DataFrame, #dataframe containing at least columns "rank" and "abundance"
                        R::Int64, #max. rank to which we sample
                        sample_name::String="downsampled"
                        )
    if R > maximum(df[:rank])
        error("available ranks < requested maximum rank")
    end
    rmax = maximum(df[:rank])
    
    if rmax == R 
        #In this case we have almost nothing to do
        return df
    end

    #Now the usual case with rmax > R:

    new_abundance = zeros(R)

    #roll out RAD into one population vector v:
    N_indivs = sum(df[:abundance])
    v = Array{Int64}(undef, N_indivs)
    N_start = 1
    for r in 1:rmax
        N_stop = N_start+df[r,:abundance]-1
        v[N_start:N_stop] .= r
        N_start = N_stop+1
    end

    Random.shuffle!(v)
    
    vpos = (N_indivs+1.0)/2.
    vtryPos = 0
    Dvpos = vpos/2.
    Rtry = 0
    unique_elems = [1]
        
    #find position in v so that elements 1...Rtry have about richness R:
        
    while Rtry != R
        vtryPos = round(Int64,vpos)
        unique_elems = unique(v[1:vtryPos])
        Rtry = length(unique_elems)
        if Rtry < R
            vpos += Dvpos
        elseif Rtry > R
            vpos -= Dvpos
        end
        Dvpos = Dvpos/2.
    end
        
    vtryPos += 1
    next_elem = v[vtryPos]
    
    while in(next_elem, unique_elems)
        vtryPos += 1
        next_elem = v[vtryPos]
    end
    vtryPos -= 1
        
    dict = countmap(v[1:vtryPos])
    
    new_abundance = sort(collect(values(dict)),rev=true)
        
    return DataFrames.DataFrame(sample = fill(sample_name, R),
                     rank = 1:R, 
                     abundance = new_abundance)
end

"""
Generates a random, non-normalized RAD following a powerlaw

Input:

- richness S (integer)

- absolute value of powerlaw exponent (float)

- optional: sample name (default: "powerlaw")

Output:

- DataFrame of RAD with columns "sample", "rank", "abundance"

"""
function random_powerlaw_RAD(S::Int64, #richness
                             k::Float64, #absolute value of exponent
                             sample_name::String="powerlaw"
                             )

    a = zeros(Int64,S) # a = abundance
    if k <= 0
        error("absolute value k of exponent must be positive")
    end
    
    # a(x) = x^(-k) #abundance a of rank x
    w = WeightVec((1:S).^(-k))
    n_ranks = 0
    while n_ranks < S #stop when richness S is reached
        i = sample(1:S,w)
        if a[i] == 0
            n_ranks += 1
        end
        a[i] += 1
    end

    return DataFrames.DataFrame(sample = fill(sample_name, S),
                     rank = 1:S, 
                     abundance = sort(a, rev=true))
end

"""
Generates a random, non-normalized RAD following a broken-stick model

Input:

- richness S (integer); the output RAD will have exactly this richness

- total number N_T of individuals used to parametrize the distribution (integer); the output RAD will in general *not* exactly have this number of individuals

- optional: sample name (default: "brokenstick")

Output:

- DataFrame of RAD with columns "sample", "rank", "abundance"

"""
function random_brokenstick_RAD(S::Int64, #richness
                                N_T::Int64,
                                sample_name::String="brokenstick"
                                )

    a = zeros(Int64,S) # a = abundance
    
    if S <= 0 || N_T <= 0
        error("S and N_T must be both positive integers")
    end

    OneOver_n = map(n -> 1.0/n, 1:S)
    c = N_T/S
    w = zeros(Float64,S)
    for i in 1:S
        w[i] = c*sum(OneOver_n[i:S])
    end
    w = WeightVec(w)
    n_ranks = 0
    while n_ranks < S #stop when richness S is reached
        i = sample(1:S,w)
        if a[i] == 0
            n_ranks += 1
        end
        a[i] += 1
    end

    return DataFrames.DataFrame(sample = fill(sample_name, S),
                     rank = 1:S, 
                     abundance = sort(a, rev=true))
end

"""
Generates the ideal broken-stick RAD for given richness S and number N_T of individuals

Input:

- richness S (integer)

- total number N_T of individuals used to parametrize the distribution (integer)

- optional: sample name (default: "brokenstick")

- optional: normalize to probability sum = 1? (default: false)
    
Output:

- DataFrame of RAD with columns "sample", "rank", "abundance"

"""
function brokenstick_RAD(S::Int64, #richness
                         N_T::Int64,
                         sample_name::String="brokenstick",
                         normalize::Bool=false
                         )

    a = zeros(S) # a = abundance
    
    if S <= 0 || N_T <= 0
        error("S and N_T must be both positive integers")
    end

    OneOver_n = map(n -> 1.0/n, 1:S)
    c = N_T/S
    for i in 1:S
        a[i] = c*sum(OneOver_n[i:S])
    end

    if normalize == true
        a = a ./ sum(a)
    end
    
    return DataFrames.DataFrame(sample = fill(sample_name, S),
                     rank = 1:S, 
                     abundance = sort(a, rev=true))
end

function RAD_rank_cutoff_normalization(
                           df::DataFrames.SubDataFrame, #dataframe containing at least columns "rank" and "abundance"
                           R::Int64 #max. rank to which we normalize (cutoff)
                           )

    if R > maximum(df[:rank])
        error("available ranks < requested cutoff rank")
    end
    rmax = maximum(df[:rank])

    if rmax == R #this is a case that is sometimes requested

        #In this case we have almost nothing to compute, except for the normalization to sum 1

        normalized_abundance = copy(df[:abundance])
        normalized_abundance ./= sum(normalized_abundance)
        
        return DataFrames.DataFrame(rank = 1:R, 
                         abundance = normalized_abundance, 
                         lower_ci = normalized_abundance, 
                         upper_ci = normalized_abundance) 
    end

    #Now the usual case with rmax > R:
    N = sum(df[:abundance])
    p = df[:abundance]/N
    normalized_abundance = p[1:R]/sum(p[1:R])
    
    return DataFrames.DataFrame(rank = collect(1:R), #ranks
                     abundance = normalized_abundance, #normalized abundances
                     lower_ci = normalized_abundance, #pseudo-lower CI
                     upper_ci = normalized_abundance) #pseudo-upper CI
end

"""
Input:

    - data frame containing RADs in long-format, with at least three columns "rank", "abundance", and "sample"

    The remaining input parameters are optional:

    - an integer number R, signaling what to do with respect to the MaxRank R:
    
        - if R = 0, find the smallest maximum rank in all RADs in the set and use that as MaxRank R

        - if R > 0, use this given R as MaxRank

    - an integer n, signaling over how many NRADs (normalized RADs) we average

        - if n = 0: use rank cutoff protocol instead of MaxRank normalization

    - a float "conf_level" for the confidence level given for the mean NRADs in the output, e.g. CI=0.9 mean that each normalized abundance we give a 90% confidence interval for the mean abundance computed by bootstrapping. The default is CI=0., i.e. no CI is computed

    - integer "n_boots" (default 100): number of bootstraps in the computation of the confidence interval. If CI=0., n_boots is ignored

    - Boolean variable "verbose": false (default) - small output; true - prints progress as normalizations proceed

    - array "procIDs" of process numbers for parallel computation. Default is [1] and no parallel computation. Generate list of such numbers for instance with addprocs(2) for parallel processing in two processes.


Output:

    - data frame of NRADs in long format with columns "rank", "abundance", "ci_lower", "ci_upper" (if CI=0., which is the default, ci_lower=ci_upper=abundance)
"""

function RAD_set_normalize(
                           df::DataFrames.DataFrame; #dataframe containing at least columns "rank", "abundance", "sample"
                           R::Int64=0, #max. rank to which we normalize: 0 for R=min(r_max), or specific R
                           n::Int64=10, #runs to be averaged over
                           conf_level::Float64=0., #confidence level, eg 0.95=95%
                           n_boots::Int64=100, #number of bootstraps for the CI evaluation
                           verbose::Bool=false, #print progress information
                           procIDs::Array{Int64,1}=[1] #array of process numbers
    )

    n_samples = length(unique(df[:sample]))
    if R == 0 #find out the minimum of all the maximum sample ranks
        max_ranks = DataFrames.by(df, :sample, d -> maximum(d[:rank]))
        R = minimum(max_ranks[:x1])
    end

    if verbose
        println("Will now MaxRank normalize ", n_samples, " RADs:")
    end

    if length(procIDs) == 1 #serial normalization of different RADs
        
        new_df = DataFrames.by(df, :sample,
                    d -> MaxRank_normalize(d, R, n, conf_level, n_boots, verbose))
     
    else #parallel normalization of different RADs
        
        from_ix, to_ix = RAD_subsets_for_procs(df, procIDs)
        n_procs = length(procIDs)
        refs = Array{Distributed.Future}(undef, n_procs)
        
        #distribute the normalization over processes, one data frame subset per process
        for i_proc in 1:n_procs
            refs[i_proc] = Distributed.@spawnat procIDs[i_proc] DataFrames.by(df[from_ix[i_proc]:to_ix[i_proc],:], :sample,
                                       d -> MaxRank_normalize(d, R, n, conf_level, n_boots, verbose))
        end
        
        #collect the normalized data frame subsets in one data frame
        new_df = fetch(refs[1])
        for i_proc in 2:n_procs
            new_df = vcat(new_df, fetch(refs[i_proc]))
            if verbose
                println("process with ID ", procIDs[i_proc], " finished")
            end
        end
    end 
    
    #sort data frame according to sample (necessary for later analyses to have a reliable order):
    if verbose
        println("sorting NRADs by sample")
    end
    sort!(new_df, [:sample])
    
    if verbose
        println("\nMaxRank normalization of ", n_samples, " RADs done.\n")
    end
    
    return new_df #dataframe of normalized RADs
end

function RAD_set_to_distmat(df::DataFrames.DataFrame, dist::Any)

    #reformat long dataframe to wide to array
    x = DataFrames.unstack(df, :rank, :sample, :abundance)
    nx = length(x)
#    sample_names = map(s -> string(s), names(x))[2:nx]

    x = convert(Array, x[:,2:nx])

    #compute distances between all pairs of RADs
    return Distances.pairwise(dist, x) #, sample_names
end

function RAD_set_cMDS(df::DataFrames.DataFrame, d::Array{Float64,2}, dim::Int64=2, accumulate_pos_eigenvalues::Bool=true)

    #compute eigenvalues
    G = dmat2gram(d)
    E = LinearAlgebra.eigen!(Symmetric(G))

    #do classical MDS:
    cMDS = classical_mds(d, dim)

    #add column for sample names for identification to MDS points
    output_df = hcat(DataFrames.DataFrame(sample = unique(df[:sample])), convert(DataFrames.DataFrame, cMDS'))
    
    eigenvals = sort(E.values[:,1],rev=true)

    #return cMDS array and eigenvalue vector descending from largest
    if accumulate_pos_eigenvalues == false
        return output_df, eigenvals
    else
        eigenvals = ifelse.(eigenvals .> 0, eigenvals, 0.0)
        eigenvals = eigenvals ./ sum(eigenvals)
        return output_df, cumsum(eigenvals)
    end
end

function RAD_kmeans_cMDS(cMDS::DataFrames.DataFrame, k::Int64)
  dims = length(cMDS)
  clust_result = kmeans(convert(Array,(cMDS[:,2:dims]))',k)
  output_df = hcat(cMDS, DataFrames.DataFrame(cluster = clust_result.assignments))
#  sort!(output_df, cols = [:cluster])
  return output_df, clust_result
end

"""
Clusters NRADs based on a distance matrix. Information about cluster names are taken
from a cMDS output data frame, and cluster assignments are appended as extra column "cluster"

Input:

  - cMDS output data frame (columns :sample, :x1, :x2)

  - NRAD-NRAD distance matrix (see RAD_set_to_distmat)

  - number of clusters k

  - optionally: sort samples according to cluster number. Note that sorting does only
  affect the data frame, not the cluster output, so that sorting will lead to inconsistent
  cluster numbering in output data frame and cluster numbering in cluster output. 
  
  - optionally further algorithmic parameters for kmedoids (package Clustering)

Output:

  - cMDS data frame with extra column "cluster"

  - kmedoids output
"""
function RAD_kmedoids(cMDS::DataFrames.DataFrame, distmat::Array{Float64,2}, k::Int64, sort::Bool=false ...)
    clust_result = kmedoids(distmat,k ...)
    if sort
        sorted = copy(clust_result.assignments)
        clust_ids = unique(sorted)
        for i in 1:k
            sorted[clust_result.assignments .== clust_ids[i]] = i
        end
        output_df = hcat(cMDS, DataFrames.DataFrame(cluster = sorted))
    else
        output_df = hcat(cMDS, DataFrames.DataFrame(cluster = clust_result.assignments))
    end
    
    return output_df, clust_result
end

#cluster RADs by affinity propagation
function RAD_set_affinityprop(df::DataFrames.DataFrame, cMDS::DataFrames.DataFrame, distmat::Array{Float64,2}, f::Float64=1.0)

    S = -distmat
    Smin = minimum(S)
    Smed = median(S)
    strength = Smin + f*(Smed-Smin)
    ncols = size(S)[1]
    S = S + strength * eye(ncols)
    clust_result = affinityprop(S)
    output_df = hcat(cMDS, DataFrames.DataFrame(cluster = clust_result.assignments))
#    sort!(output_df, cols = [:cluster])
    
    return output_df, S, clust_result
end

function RAD_set_representatives(cMDSclusters::DataFrames.DataFrame, normRADs::DataFrames.DataFrame, sets::Tuple)
    n_sets = length(sets)
    n_dim =  find(names(cMDSclusters) .== :cluster)[1]-2 #this is the column of the highest dimension

    #initialize array for the representative RADs
    all_repres_RADs = DataFrames.DataFrame(vcat(eltypes(normRADs),Float64,Float64,Int64),
                                vcat(names(normRADs),:min,:max,:set), 0)

    for i in 1:n_sets
        #set_i is the part of cMDSclusters that forms the ith set
        set_i = cMDSclusters[convert(Array{Bool,1}, map(x -> in(x, sets[i]),cMDSclusters[:cluster])),:]

        #geometric center of set i
        center_i = convert(Array,aggregate(set_i[:,2:(n_dim+1)],mean))'
        coors_i = convert(Array,set_i[:,2:(n_dim+1)])'

        #represent_i is the "representative" element of the set (closest to center)
        represent_i = set_i[indmin(pairwise(SqEuclidean(),center_i,coors_i)),:]

        #retrieve the RAD corresponding to represent_i
        new_repres_RAD = copy(normRADs[normRADs[:,:sample] .== represent_i[:,:sample],:])

        #compute min and max abundances for each rank in set
        #first identify all sample ids in set:
        samples_i = cMDSclusters[convert(Array{Bool,1}, map(x -> in(x, set_i[:sample]), cMDSclusters[:sample])),:sample]

        #then determine the min for each rank in the set:
        x = DataFrames.by(normRADs[convert(Array{Bool,1},
                                map(x -> in(x, samples_i), normRADs[:sample])),:],
               :rank,
               df -> minimum(df[:abundance]))
        new_repres_RAD[:min] = x[:x1]

        #same for maximum:
        x = DataFrames.by(normRADs[convert(Array{Bool,1},
                                map(x -> in(x, samples_i), normRADs[:sample])),:],
               :rank,
               df -> maximum(df[:abundance]))
        new_repres_RAD[:max] = x[:x1]

        #add a column for the set:
        new_repres_RAD[:set] = fill(i, size(new_repres_RAD)[1])

        #append to overall data structure
        all_repres_RADs = vcat(all_repres_RADs, new_repres_RAD)
    end
    
    return all_repres_RADs
    
end

function RAD_cmds_set_means(cMDSclusters::DataFrames.DataFrame, normRADs::DataFrames.DataFrame, sets::Tuple)
    n_sets = length(sets)
    n_dim =  find(names(cMDSclusters) .== :cluster)[1]-2 #this is the column of the highest dimension

    #initialize arrays for the mean RADs
    all_mean_RADs = DataFrames.DataFrame([Int64;Float64;Float64;Float64;Int64],[:rank;:mean_abundance;:min;:max;:set], 0)
    max_rank = maximum(normRADs[:rank])
    new_mean_RAD = DataFrames.DataFrame([Int64;Float64;Float64;Float64;Int64],[:rank;:mean_abundance;:min;:max;:set], max_rank)
    
    for i in 1:n_sets
        #set_i is the part of cMDSclusters that forms the ith set
        set_i = cMDSclusters[convert(Array{Bool,1}, map(x -> in(x, sets[i]),cMDSclusters[:cluster])),:]

        
        #compute ranks, mean, min and max abundances for each rank in set
        #first identify all sample ids in set:
        samples_i = cMDSclusters[convert(Array{Bool,1}, map(x -> in(x, set_i[:sample]), cMDSclusters[:sample])),:sample]

        new_mean_RAD[:rank] =
            unique(normRADs[convert(Array{Bool,1}, map(x -> in(x, samples_i), normRADs[:sample])),:rank])

        #then determine the mean for each rank in the set:
        x = DataFrames.by(normRADs[convert(Array{Bool,1},
                                map(x -> in(x, samples_i), normRADs[:sample])),:],
               :rank,
               df -> mean(df[:abundance]))
        new_mean_RAD[:mean_abundance] = x[:x1]

        #determine the min for each rank in the set:
        x = DataFrames.by(normRADs[convert(Array{Bool,1},
                                map(x -> in(x, samples_i), normRADs[:sample])),:],
               :rank,
               df -> minimum(df[:abundance]))
        new_mean_RAD[:min] = x[:x1]

        #same for maximum:
        x = DataFrames.by(normRADs[convert(Array{Bool,1},
                                map(x -> in(x, samples_i), normRADs[:sample])),:],
               :rank,
               df -> maximum(df[:abundance]))
        new_mean_RAD[:max] = x[:x1]

        #add a column for the set:
        new_mean_RAD[:set] = fill(i, size(new_mean_RAD)[1])

        #append to overall data structure
        all_mean_RADs = vcat(all_mean_RADs, new_mean_RAD)
    end
    
    return all_mean_RADs
    
end

"""
Average a set of RADs that have the same richness.

Input:

- RADs: DataFrame of set of RADs of equal richness

- name (optional): a "sample" name for the averaged RAD (default: "mean")

- normalize (optional): whether to normalize the mean RAD to total probability = 1 (default: false)

- conf_level (optional): confidence level (default: 0.9, i.e. from 0.05 quantile to 0.95 quantile)


Output:

- DataFrame of mean RAD or average RAD with the usual columns of NRADs (sample, rank, abundance, lower_ci, upper_ci)

"""
function RAD_set_mean(
                      RADs::DataFrames.DataFrame;
                      name::String="mean",
                      normalize::Bool=false,
                      conf_level::Float64=0.90, #confidence level
                      n_boots::Int64=100
                      )

    #first check whether all RADs have same richness
    richness = DataFrames.by(RADs, :sample, x -> maximum(x[:rank]))[:x1]
    if minimum(richness) != maximum(richness)
        error("RADs do not have same richness")
    end

    meanRAD = hcat(DataFrames.DataFrame(sample = fill(name,richness[1])),
                   DataFrames.by(RADs, :rank, df -> mean(df[:abundance])))
    rename!(meanRAD, :x1 => :abundance)


    q_lower = 0.5*(1.0-conf_level) #lower quantile of CI
    q_upper = 1.0-q_lower #upper quantile of CI

    ci_lower = zeros(richness[1])
    ci_upper = zeros(richness[1])
    for i in 1:richness[1]
        ci_lower[i], ci_upper[i] =
            bootstrap_mean_CI(RADs[RADs[:rank] .== i, :abundance],
                              q_lower,
                              q_upper,
                              n_boots)
    end 
    if normalize == true
        sum_mean_RAD = sum(meanRAD[:abundance])
        meanRAD[:abundance] = meanRAD[:abundance] ./ sum_mean_RAD
        ci_lower = ci_lower ./ sum_mean_RAD
        ci_upper = ci_upper ./ sum_mean_RAD
    end

    meanRAD[:lower_ci] = ci_lower
    meanRAD[:upper_ci] = ci_upper
    
    return meanRAD
    
end

"""
Input:
            
    - array x of values of which we bootstrap the mean

    - q_lower: lower quantile

    - q_upper: upper quantile

    - n_boots: number of bootstraps

Output:

    - ci_lower

    - ci_upper
            
"""

function bootstrap_mean_CI(x::AbstractArray{Float64},
                           q_lower::Float64=0.05,
                           q_upper::Float64=0.95,
                           n_boots::Int64=100)
    sample_size = length(x)
    means = zeros(n_boots)
    for i in 1:n_boots
        means[i] = mean(sample(x, sample_size))
    end
    ci_lower = quantile(means, q_lower)
    ci_upper = quantile(means, q_upper)

    return ci_lower, ci_upper
end

"""
Coarsen the log10-rank dimension to n pseudo-log10-ranks, usually for feature reduction as
preparation to learning a model.

Input:

    - 2D array of RADs, 1 RAD per row, 1 rank per column

    - number n of pseudo-log10-ranks

Output:

    - 2D array of pseudo-log10-ranks, 1 pseudo-log10-abundance per row, 1 pseudo-rank per column. The pseudo-log10-abundance is computed as log10 of mean of abundances for a range of ranks.

    - 'lowers': lower rank boundary of each pseudo-rank (1D int array)

    - 'uppers': upper rank boundary of each pseudo-rank (1D int array)

"""
function RAD_log10log10coarser(features::Array{Float64,2}, n::Int64)
    
    #features: 1 row per RAD (or sample), 1 column per rank
    
    #transpose for efficiency (columnwise processing possible)
    features_t = features'
    
    R, n_samples = size(features_t)

    #a (new) pseudo-log10-rank i aggregates abundances between
    #ranks lowers[i] and uppers[i]
    lowers = zeros(Int64,n)
    uppers = zeros(Int64,n)
    lg_features_t = zeros(n, n_samples);
    
    lg_breaks = collect(1:n).*(log10(R)/n)

    #first determine the breaks along the pseudo-log10-rank axis
    r = 1
    for i in 1:n
        lowers[i] = r
        while(log10(r)<=lg_breaks[i])
            r += 1
        end
        uppers[i] = maximum([lowers[i];r-1])
    end

    #second do the averaging of abundances for each pseudo-log10-rank
    for j in 1:n_samples
        for i in 1:n
            lg_features_t[i,j] = log10(mean(features_t[lowers[i]:uppers[i],j]))
        end
    end
    
    return lg_features_t', lowers, uppers
    #return again 1 row per RAD (or sample), 1 column per coarsened rank
end

function RAD_leave_one_out_regression_forest(
    variable::Array{Float64}, 
    features::Array{Float64,2},
    forest_pars...)
    
    n = length(variable)
    all_i = collect(1:n)
    
    prediction = zeros(n)
    
    for i in 1:n
        if i%10 > 0 # print to inform user about progress
            print(".")
        else
            print(i)
        end
        ixs = deleteat!(collect(1:n),i) 
        model_i = 
          build_forest(variable[ixs], features[ixs,:], forest_pars...)
        prediction[i] = apply_forest(model_i, features[i,:][:])
    end
    
    return cor(variable, prediction), prediction
end

"""
Input:
        - NRAD array (columns are ranks, rows are samples)
        
        - name to appear in the data frame

        - percentile, interval [percentile, 100-percentile] is reported to indicate the spread of the abundance values of each rank

Output:
        - data frame with columns rank, abundance (=median), pi_lower, pi_upper (percentile interval = pi), name
        
"""
function RADs_median(
    RADs::Array{Float64,2}, # columns are ranks, rows are samples
    name::String, # name to appear in data frame
    perc::Float64=25. #percentile; the interval [perc,100-perc]
                      #is reported
    ) 

    n, R = size(RADs)
≥
    medians = vec(median(RADs,1))

    perc_upper = 100.0-perc
    
    pi_lower = map(j -> percentile(RADs[:,j], perc), 1:R)
    pi_upper = map(j -> percentile(RADs[:,j], perc_upper), 1:R)
    
    return DataFrames.DataFrame(
        rank = collect(1:R),
        abundance = medians,
        pi_lower = pi_lower,
        pi_upper = pi_upper,
        name = fill(name, R))
end

"""
Input:
        - NRAD array (columns are ranks, rows are samples)
        
        - name to appear in the data frame

        - confidence level for bootstrapped confidence interval (default: conf_level=0.9)

        - number of bootstraps for the estimation of the mean and its confidence interval (default: n_boots=100)

Output:
        - data frame with columns rank, abundance (=average), lower_ci, upper_ci
        
"""
function RADs_average(
                      RADs::Array{Float64,2}, # columns are ranks, rows are samples
                      name::String, # name to appear in data frame
                      conf_level::Float64=0.9, #confidence level for bootstrapped confidence interval
                      n_boots::Int64=100 #number of bootstraps
                      ) 

    n, R = size(RADs)
    ci_lower = zeros(R)
    ci_upper = zeros(R)
    
    average = vec(mean(RADs,1))

    q_lower = 0.5*(1.0-conf_level)
    q_upper = 1.0-q_lower
    
    for j in 1:R
        #for each rank we bootstrap the means
        ci_lower[j], ci_upper[j] = bootstrap_mean_CI(RADs[:,j], q_lower, q_upper, n_boots)
    end
    
    return DataFrames.DataFrame(
        rank = collect(1:R),
        abundance = average,
        lower_ci = ci_lower,
        upper_ci = ci_upper,
        name = fill(name, R))
end


"""
For given pairs (x,y) (with x>0) splits x-range into n_bins log-equal bins (same size of each bin on log-scale), and computes average x and y values in each bin, together with confidence intervals assuming a normal distribution.
        
Input:

    - 1D-float array x

    - 1D-float array y

    - number n_bins of bins

    - optionally: confidence level (default 0.95)

    - optionally: xmin (lower limit of first bin); if zero (default) takes the minimum x value

    - optionally: xmax (upper limit of last bin); if zero (default) takes the maximum x value

Output:

    - Data frame of means of x and y per bin, and confidence interval (ci_lower, ci_upper) in each bin
"""    
function log10binmeans(
    x::Array{Float64,1}, 
    y::Array{Float64,1}, 
    n_bins::Int64,
    conf_level::Float64=0.95,
    xmin::Float64=0.,
    xmax::Float64=0.
    )
    
    if xmin == 0.
        xmin = minimum(x)
    end
    
    if xmax == 0.
        xmax = maximum(x)
    end
    
    ci_factor = quantile(Normal(), 1.0-(1.0-conf_level)/2.0)
    
    xminlog10 = log10(xmin)
    xmaxlog10 = log10(xmax)
    
    #split the interval between lg(xmin) and log(xmax)
    #in n_bins bin (having n_bins+1 boundaries)
    xbins = 10.0.^range(xminlog10, stop=xmaxlog10, length=n_bins+1)
    
    xmeans = zeros(n_bins)
    ymeans = zeros(n_bins)
    ci_lower = zeros(n_bins)
    ci_upper = zeros(n_bins)
    
    sp = sortperm(x)
    xsort = x[sp]
    ysort = y[sp]
    jstop = length(xsort)+1
    
    j = 1
    while xsort[j] < xmin
        j += 1
    end
    for i in 1:n_bins
        jmin = j
        while (j < jstop) && (xsort[j] < xbins[i+1])
            j+=1
        end
        xmeans[i] = mean(xsort[jmin:(j-1)])
        ymeans[i] = mean(ysort[jmin:(j-1)])
        ci_lower[i] = ymeans[i] - ci_factor * sem(ysort[jmin:(j-1)])
        ci_upper[i] = ymeans[i] + ci_factor * sem(ysort[jmin:(j-1)])
        if j == jstop
            break
        end
    end
    
    return DataFrames.DataFrame(
            xmeans = xmeans, ymeans = ymeans, 
            ci_lower = ci_lower, ci_upper = ci_upper
            )
end

"""
Transforms wide-format OTU tables into long-format RADs.
        
Input:

    - OTU table as wide data frame, one column per sample (column names = sample names).

    - optionally: skip (default: skip = 0); sometimes the first columns contains OTU names etc. If you want to skip this column (or any number of columns), give skip the corresponding value, e.g. set it to 1 to skip the first column.
        
Output:

    - RADs as long data frame with three columns: rank, abundance (not normalized), sample.
"""    
function RAD_set_from_OTU_table(OTUtable::DataFrames.DataFrame, skip::Int64=0)
    n_columns = length(OTUtable) 
    RADs = 
        DataFrames.DataFrame(
            rank = Array{Float64,1},
            abundance = Array{Float64,1},
            sample = Array{String,1}
        )
    for i in (skip+1):n_columns
        abundance = sort(OTUtable[OTUtable[:,i] .!= 0,i], rev=true)
        rank = collect(1:length(abundance))
        sample = fill(string(names(OTUtable)[i]), length(abundance))
        if i == (skip+1)
            RADs = DataFrames.DataFrame(rank = rank, abundance = abundance, sample = sample)
        else
            RADs = vcat(RADs, DataFrames.DataFrame(rank = rank, abundance = abundance, sample = sample))
        end
    end
    return RADs
end

"""
Plots a set of RADs with log-log-scaling and 1 color per sample.

Input:
        - data frame with at least the columns 'rank', 'abundance' and 'sample'
        
        - optionally: plot confidence intervals (if present in data frame) as shaded areas
        around NRADs 

Output:
        - plot
"""
function RAD_set_plot(RADs::DataFrames.DataFrame, plot_CIs::Bool=false)
    if plot_CIs == true
        return StatPlots.@df RADs Plots.plot(:rank, :abundance, xaxis=("rank",:log10), yaxis=("abundance",:log10), ribbon=(:abundance-:lower_ci,:upper_ci-:abundance), group=:sample)
    else
        return StatPlots.@df RADs Plots.plot(:rank, :abundance, group=:sample, xaxis=(:log10, "rank"), yaxis=(:log10, "abundance"))
    end
end

# function RAD_set_plot(RADs::DataFrame, plot_CIs::Bool=false, add_points::Bool=false)
#     if plot_CIs
#         if add_points
#             return plot(RADs, x = "rank", y = "abundance", color = "sample", 
#                         Geom.line, Geom.point,
#                         ymin = "lower_ci", ymax = "upper_ci", Geom.ribbon,
#                         Scale.x_log10, Scale.y_log10)
#         else
#             return plot(RADs, x = "rank", y = "abundance", color = "sample", 
#                         Geom.line, 
#                         ymin = "lower_ci", ymax = "upper_ci", Geom.ribbon,
#                         Scale.x_log10, Scale.y_log10)
#         end
#     else
#         if add_points
#             return plot(RADs, x = "rank", y = "abundance", color = "sample", 
#                         Geom.line, Geom.point,
#                         Scale.x_log10, Scale.y_log10)
#         else 
#             return plot(RADs, x = "rank", y = "abundance", color = "sample", 
#                         Geom.line, 
#                         Scale.x_log10, Scale.y_log10)
#         end
#     end
# end

"""
Computes Shannon entropies -sum p_i log p_i (in nats) for a set of RADs or NRADs.

Input:
        - data frame of set of RADs or NRADs with at least the columns 'abundance' and 'sample'

Output:
        - data frame with columns 'sample' and 'entropy'
"""
function RAD_set_entropies(RADs::DataFrames.DataFrame)
    return rename(DataFrames.by(RADs, :sample, d -> StatsBase.entropy(d[:abundance])), :x1 => :entropy)
end

"""
Computes for a set of RADs or NRADs Shannon evenness H/ln S (method argument :Shannon) or
    Heips evenness e^H/S (method argument :Heip).

Input:
        - data frame of set of RADs or NRADs with at least the columns 'abundance' and 'sample'

Output:
        - data frame with columns 'sample' and 'evenness'
"""
function RAD_set_evenness(RADs::DataFrames.DataFrame, method::Symbol=:Shannon)
    if method == :Shannon
        return rename(DataFrames.by(RADs, :sample, d -> StatsBase.entropy(d[:abundance])/log(length(d[:abundance]))), :x1 => :evenness)
    elseif method == :Heip
        return rename(DataFrames.by(RADs, :sample, d -> exp(StatsBase.entropy(d[:abundance]))/length(d[:abundance])), :x1 => :evenness)
    else
        error("evenness method not implemented")
    end
end

"""
Identify the minimum richness among a set of RADs.

Input:
    - data frame with RADs (or NRADs) with at least columns :sample and :rank

Output:
    - minimum richness
"""
function RAD_set_minimum_richness(RADs::DataFrames.DataFrame)
    max_ranks = DataFrames.by(RADs, :sample, d -> maximum(d[:rank]))
    return minimum(max_ranks[:x1])
end


"""
Report richness values (maximum ranks) for a set of RADs.

Input:
    - data frame with RADs (or NRADs) with at least columns :sample and :rank

Output:
    - data frame with columns :sample and :richness
"""
function RAD_set_richness(RADs::DataFrames.DataFrame)
    max_ranks = DataFrames.by(RADs, :sample, d -> maximum(d[:rank]))
    return rename!(max_ranks, :x1 => :richness)
end

"""
Report the summed abundances per RAD in a set of RADs. For non-normalized RADs this corresponds to the number of sampled individuals, for NRADs this sum should be 1.

Input:
    - data frame with RADs (or NRADs) with at least columns :sample and :rank

Output:
    - data frame with columns :sample and :abundance_sum
"""
function RAD_set_abundance_sum(RADs::DataFrames.DataFrame)
    abundance_sums = DataFrames.by(RADs, :sample, d -> sum(d[:abundance]))
    return rename!(abundance_sums, :x1 => :abundance_sum)
end

"""
Tests whether abundance columns of two RADs are approximately equal (using the isapprox function of julia).

Input:

    - two data frames RAD_1 and RAD_2 with at least the column "abundance"

Output:

    - true if abundance columns are approximately numerically the same
"""
function test_approx_abundance_equality(RAD_1::DataFrames.DataFrame, RAD_2::DataFrames.DataFrame)
    if length(RAD_1[:abundance]) != length(RAD_2[:abundance])
        println("different lengths of abundance arrays")
        return false
    else
        for i in 1:length(RAD_1[:abundance])
            if !isapprox(RAD_1[i,:abundance], RAD_2[i,:abundance])
                println("differences at rank $i")
                return false
            end
        end
    end
    return true 
end

"""
        Splits up a RAD data frame into chunks of roughly equal size, one chunk per process.
        Used e.g. for parallel normalization of RADs.

        Input:
        
        - data frame df of RADs (minimum columns: sample, rank, abundance)
        
        - array of process IDs (typically small integer numbers)


        Output:
        
        - a set from_ix of starting indexes for each RAD subset
            
        - a set to_ix of terminal indexes for each RAD subset
        
"""
function RAD_subsets_for_procs(
    df::DataFrames.DataFrame, #RADs
    procIDs::Array{Int64,1} #process numbers (IDs)
    )

    n_procs = length(procIDs)
    from_ix = zeros(Int64, n_procs)
    to_ix = zeros(Int64, n_procs)
    
    samples = unique(df[:sample])
    n_last = length(samples)+1
    
    chunk_bounds_float = collect(range(1, stop=n_last, length=n_procs+1))
    
    for chunk in 1:n_procs

        first_in_chunk = 
        convert(Int64, round(chunk_bounds_float[chunk]))
        from_ix[chunk] = 
        findfirst(df[:sample] .== samples[first_in_chunk])
        
        last_in_chunk = 
        convert(Int64, round(chunk_bounds_float[chunk+1])-1)
        to_ix[chunk] = 
        findlast(df[:sample] .== samples[last_in_chunk])
        
    end
    
    return from_ix, to_ix
end

"""
Extract rows from a RAD data frame that belong to a subset of samples.

Input:

     - RAD (or NRAD) data frame, should at least have a "sample" column.

     - subset of sample names, e.g. ["IgG_A";"IgG_B"]

Output:

     - data frame consisting of the subset of samples.
            
"""
function RAD_samples_subset(df::DataFrames.DataFrame, subset::Array{String})
    subset = convert(Array{Bool}, map(x -> in(x, subset), df[:sample]))
    return df[subset,:]
end


"""
Extract rows from a RAD data frame that belong to a subset of samples with names consistent with array of regular expressions

Input:

     - RAD (or NRAD) data frame, should at least have a "sample" column.

     - array of regular expressions, e.g. ["IgG_[A-D]\\\$";"IgM_*"]

Output:

     - data frame consisting of the subset of samples.
            
"""
function RAD_samples_regex_set(df::DataFrames.DataFrame, subset::Array{String})    
    n = length(subset)
    re_set = map(i -> Regex(subset[i]), 1:n)
    subset = convert(Array{Bool}, map(x -> is_in_regex_set(x, re_set), df[:sample]))
    return df[subset,:]
end

function is_in_regex_set(x::String, re_set::Array{Regex})
    n = length(re_set)
    for i in 1:n
        if occursin(re_set[i],x)
            return true
        end 
    end
    return false 
end


"""
Computes a summary for a set of RADs.

Input:

    - set of RADs as data frame with at least columns "sample", "rank", "abundance"

Output:

    - data frame with various summary information
    
"""
function RAD_set_summary(df::DataFrames.DataFrame)
    n_categories = 16
    summary = DataFrames.DataFrame(quantity =
                                    ["number of RADs",
                                     "richness minimum",
                                     "richness median",
                                     "richness maximum",
                                     "abundance sum minimum",
                                     "abundance sum median",
                                     "abundance sum maximum",
                                     "abundance minima minium",
                                     "abundance minima median",
                                     "abundance minima maximum",
                                     "abundance medians minimum",
                                     "abundance medians median",
                                     "abundance medians maximum",
                                     "abundance maxima minimum",
                                     "abundance maxima median",
                                     "abundance maxima maximum"],
                        value = Array{Any}(undef, n_categories))

    summary[1,2] = length(unique(df[:sample]))
    
    richness = DataFrames.by(df, :sample, x -> maximum(x[:rank]))[:x1]
    summary[2,2] = minimum(richness)
    summary[3,2] = Statistics.median(richness)
    summary[4,2] = maximum(richness)

    abundance_sum = DataFrames.by(df, :sample, x -> sum(x[:abundance]))[:x1]
    summary[5,2] = minimum(abundance_sum)
    summary[6,2] = Statistics.median(abundance_sum)
    summary[7,2] = maximum(abundance_sum)

    #minima of abundance
    aggregated = DataFrames.by(df, :sample, x -> minimum(x[:abundance]))[:x1]
    summary[8,2] = minimum(aggregated)
    summary[9,2] = Statistics.median(aggregated)
    summary[10,2] = maximum(aggregated)

    #Statistics.medians of abundance
    aggregated = DataFrames.by(df, :sample, x -> Statistics.median(x[:abundance]))[:x1]
    summary[11,2] = minimum(aggregated)
    summary[12,2] = Statistics.median(aggregated)
    summary[13,2] = maximum(aggregated)

    #maxima of abundance
    aggregated = DataFrames.by(df, :sample, x -> maximum(x[:abundance]))[:x1]
    summary[14,2] = minimum(aggregated)
    summary[15,2] = Statistics.median(aggregated)
    summary[16,2] = maximum(aggregated)

    return summary
end

"""
Reads a tabular file (typically .csv or .csv.gz) with at least columns "sample", "rank", "abundance" and converts "sample" column into String if necessary. Returns tested file, if necessary with converted sample column.

Input:

    - file name of tabular file

Output:

    - tested RAD data as data frame
 
"""
function read_RAD_set(filename::AbstractString)
    df = CSV.read(GZip.open(filename))
    Names = map(x->string(x), names(df))

    if !in("sample", Names)
        error("\"sample\" column missing in RAD file")
    elseif !in("rank", Names)
        error("\"rank\" column missing in RAD file")
    elseif !in("abundance", Names)
        error("\"abundance\" column missing in RAD file")
    end
        
    #we use the first column with name "sample" as sample names
    ix_sample = findfirst(string(:sample).==Names)

    if typeof(df[:,ix_sample])!=Array{String,1}
        df[:sample] = convert(Array{String}, map(x->string(x), df[:sample]))
    end

    return df 
end

"""
Takes a RAD set data frame, splits it into a number of approximately equally sized subsets (in terms of number of RADs), and writes a file for each subset.

Input:

    - data frame of RAD set

    - stem name of output files; e.g. stem name "myRADs" will produce files myRADs-1.csv.gz, myRADs-2.csv.gz, etc.

    - number of files (Int64) into which the RAD set will be split

Output:

    - writes the files to disk
    
"""
function split_RAD_set_into_files(rads::DataFrames.DataFrame, stem_name::AbstractString, n_files::Int64)
    
    samples = unique(rads[:sample])
    n_samples = length(samples)
    rads_per_batch = round(Int64, n_samples / n_files)
    start_batch = 0
    end_batch = 0
    for i in 1:(n_files-1)
        start_batch = (i-1)*rads_per_batch + 1
        end_batch = i*rads_per_batch
        radsub = RAD_samples_subset(rads, convert(Array{String},samples[start_batch:end_batch]))
        fh = GZip.open(string(stem_name, "-", i, ".csv.gz"),"w")
        CSV.write(fh, radsub)
        GZip.close(fh)
    end
    
    #write the rest
    start_batch = end_batch+1
    end_batch = n_samples
    radsub = RAD_samples_subset(rads, convert(Array{String},samples[start_batch:end_batch]))
    fh = GZip.open(string(stem_name, "-", n_files, ".csv.gz"),"w")
    CSV.write(fh, radsub)
    GZip.close(fh)
end 

"""
Reformats a wide NRAD data frame to long format data frame.

Input:

    - NRAD matrix (1 NRAD per row, 1st column is sample) 

Output:

    - NRAD data frame with columns sample, rank, and abundance.
"""

function RAD_set_from_matrix(df_wide::DataFrames.DataFrame)
    samples = convert(Array{String},map(x->string(x),df_wide[:,1]))
    n_samples = length(samples)
    R = size(df_wide)[2]-1 #MaxRank

    return DataFrames.DataFrame(
                     sample = repeat(samples, inner=[R]), 
                     rank = repeat(collect(1:R), outer=[n_samples]),
                     abundance = vec(convert(Array{Float64,2},df_wide[:,2:end])')
                     )
end

"""
Transform a non-normalized RAD into a population vector, i.e. roll the RAD out.

Input:

- RAD data frame with at least columns "rank" and "abundance".

Output:

- Population vector (typically very long).

"""

function RAD_to_population_vector(RAD::DataFrames.DataFrame)
    rmax = maximum(RAD[:rank])
    N_indivs = sum(RAD[:abundance])
    v = Array(Int64, N_indivs) #population vector v
    N_start = 1
    for r in 1:rmax
        N_stop = N_start+RAD[r,:abundance]-1
        v[N_start:N_stop] = r
        N_start = N_stop+1
    end
    return v
end

"""
Scales the abundance values in a RAD set down so that abundances in each RAD are probabilities (sum = 1). Important: In contrast to the MaxRank normalization, this function does not reduce the ranks of the RADs.

Input:

- Set of non-normalized RADs with at least columns "sample", "rank", and "abundance".

Output:

- RADs with abundances turned into probabilities. (Re-)Sorted according to "sample" and "rank".

"""
function RAD_set_abundance_normalization(RADs::DataFrames.DataFrame)
    rads = copy(RADs)
    sort!(rads, cols = [:sample, :rank])
    rads[:abundance] = DataFrames.by(rads, :sample, df -> df[:abundance] ./ sum(df[:abundance]))[:x1]
    return rads
end

end # module



