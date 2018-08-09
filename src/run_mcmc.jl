include("./obs_funcs.jl")
include("./fit_funcs.jl")
include("./utils.jl")

using CSV
using AffineInvariantMCMC
using CenQuenUtils
import Jackknife.get_sdss_subvols

# argument passed is number sample to run mcmc for
num_sample = int(ARGS[2])

# take the corresponding data
datdf, vols = load_groupdat(num_sample)

# compute jackknife volumes of the data points
radecs = zeros(size(datdf)[1], 2)
radecs[:, 1] = rad2deg.(datdf[:ra])
radecs[:, 2] = rad2deg.(datdf[:dec])

datdf[:jackvol] = get_sdss_subvols(radecs)

# take the mock with masses and environment for modeling
if num_sample == 1
    mockdf = CSV.read([2])
else if num_sample == 2
    mockdf = CSV.read([2])
else if num_sample == 3
    mockdf = CSV.read([2])
else
    println("incorrect argument")
end

## add extra columns for modeling
mockdf[:logMh] = log10.(mockdf[:halo_mvir])
mockdf[:Mdot] = (mockdf[:halo_mvir] .- mockdf[:halo_mz_1]) ./ mockdf[:halo_mz_1]
mockdf[:Mdot][mockdf[:halo_mz_1] .== 0] = 1
mockdf[:logMd] = log10.(mockdf[:Mdot] + 1)
mockdf[:logc] = log10.(mockdf[:halo_nfw_conc])
mockdf[:logspin] = log10.(mockdf[:halo_spin])
mockdf[:logρ] = log10.(mockdf[:delta] + 1)
mockdf[:logRe] = ones(mockdf[:log10M]) * mean(datdf[:logRe])
mockdf[:logsurf] = ones(mockdf[:log10M]) * mean(datdf[:logsurf])
mockdf[:logv] = ones(mockdf[:log10M]) * mean(datdf[:logv])

# TODO
# TODO figure out how to pass the variables as arguments to script!!
# TODO

# Define the bin edges of the galaxy properties
#mbins = [9.4, 9.6, 9.8, 10.1, 10.3, 10.45, 10.6, 10.8, 11.0]
mbins = [9.4, 9.6, 9.8, 10.0, 10.2, 10.4, 10.6, 10.8, 11.0]
#mbins = [9.4, 9.8, 10.3, 10.6, 11.0]
#dbins = linspace(-0.5, 1, 9)
dbins = linspace(-0.8, 1.0, 9)
const rbins = [-3, -2.75, -2.5, -2.25, -2, -1.75]
const sbins = [8.5, 8.75, 9.0, 9.25, 9.5]
const vbins = [1.9, 2.05, 2.2, 2.35, 2.5]
#const rbins = [minimum(datdf[:logRe]), maximum(datdf[:logRe]) + 1]
#const sbins = [minimum(datdf[:logsurf]), maximum(datdf[:logsurf]) + 1]
#const vbins = [minimum(datdf[:logv]), maximum(datdf[:logv]) + 1]

# the columns that matter for modeling
datcols = [:log10M, :logρ, :logRe, :logsurf, :logv]

# separate data into dataframes of mass and environment
nmbins = length(mbins) - 1
ndbins = length(dbins) - 1

deltadat = Array{Array{DataFrame, 1}}(nmbins)

for i in 1:nmbins
    deltadat[i] = Array{DataFrame}(ndbins)
    massdf = datdf[mbins[i] .<= datdf[:log10M] .< mbins[i + 1], datcols]
    for j in 1:ndbins
        deltadat[i][j] = massdf[dbins[j] .<= massdf[:logρ] .< dbins[j + 1], datcols]
    end
end

# create the same for mocks
mockcols = [:logMh, :logMd, :logc, :logspin, :log10M, :logρ, :logRe, :logsurf, :logv]

const mockdfarr = Array{Array{DataFrame, 1}}(nmbins)

for i in 1:nmbins
    mockdfarr[i] = Array{DataFrame}(ndbins)
    massdf = mockdf[mbins[i] .<= mockdf[:log10M] .< mbins[i + 1], mockcols]
    for j in 1:ndbins
        mockdfarr[i][j] = massdf[dbins[j] .<= massdf[:logρ] .< dbins[j + 1], mockcols]
    end
end

# use these to make the data observation
const obsdat = bincounts(deltadat, rbins, sbins, vbins)

# create a mask of the nonzero elements of the observations
const nonzmask = find(obsdat .> 0)

# and a tuple of the sums of the data and the log of the data
const obsums = (sum(obsdat), sum(obsdat[nonzmask] .* (log.(obsdat[nonzmask]))))

# set the volume ratio
const volfac = vols[num_sample] / 400^3

# add method to model bincount generation
gen_mod_bincounts(params::Array{Float64,1}) = gen_mod_bincounts(params, mockdfarr, rbins, sbins, vbins)

# define the log likelihood for the sampling
function like(params::Array{Float64,1})

    mockhist = gen_mod_bincounts(params)
    mockhist = mockhist .* volfac + exp(-125)

    score = cstat(mockhist, obsdat, obsums, nonzmask)
    -score
end
