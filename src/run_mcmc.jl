include("./obs_funcs.jl")
include("./fit_funcs.jl")
include("./utils.jl")

using CSV
using Glob
using AffineInvariantMCMC
using .CenQuenUtils

import BlackBoxOptim
import .Jackknife.get_sdss_subvols


"""
This function returns the string for the base data directory
"""
function get_datdir()
    datdir = join(split(realpath(@__DIR__), "/")[1:(end - 1)], "/")
    datdir = string(datdir, "/dat")
    return datdir
end


"""
Convert the parameter config integer to an array of zeros and ones
representing the different parameters.
"""
function get_vars(var_int::Int)
    return [parse(Int, n) for n in bin(var_int, 3)]
end


"""Compute filename strings from the sample number and variable switch"""
function get_thres_varstring(sample_int::Int, varswitch::Array{Int})

    ths = ["18", "19", "20a", "20b"]
    th = ths[sample_int]

    vars = ["Re", "surf", "vdisp"]
    varstring = join(vars[Bool.(varswitch)], "_")

    return th, varstring
end


"""Add method for computing the strings directly from the config integers"""
get_thres_varstring(sample_int::Int, varnum::Int) = get_thres_varstring(sample_int, get_vars(varnum))


"""
This function loads the correct NYU-VAGC data sample based on which mass range is being analysed
"""
function get_dat(data_sample::Int)

    if data_sample in 1:3
        datdf, vols = load_groupdat(data_sample)
        volume = vols[data_sample]
    elseif data_sample == 4
        datdf, vols = load_groupdat(3)
        volume = vols[3]
    else
        throw(ArgumentError("data_sample must be an integer from 1-4"))
    end

    # compute the jackknife subvolumes of the data
    radecs = zeros(size(datdf)[1], 2)
    radecs[:, 1] = rad2deg.(datdf[:ra])
    radecs[:, 2] = rad2deg.(datdf[:dec])

    datdf[:jackvol] = get_sdss_subvols(radecs)

    return datdf, volume
end


function get_means(data_sample::Int, datdf)

    minmass = [9.4, 9.8, 10.3, 10.6][data_sample]
    maxmass = [9.8, 10.3, 10.6, 11.0][data_sample]

    bindf = datdf[minmass .<= datdf[:log10M] .< maxmass, :]

    return mean(bindf[:logRe]), mean(bindf[:logsurf]), mean(bindf[:logv])
end


"""
This function loads the mock that has previously been generated to match the
stellar mass and environment density distributions of the corresponding data sample.
"""
function get_mock(data_sample::Int, meanr, means, meanv)

    # load the corresponding mock
    ths = [18, 19, 20, 20]
    th = ths[data_sample]
    datdir = get_datdir()
    mockdf = CSV.read(string(datdir, "/mocks/M", th, "_cenquen_mock.csv"))

    ## add extra columns for modeling
    mockdf[:logMh] = log10.(mockdf[:halo_mvir])
    mockdf[:Mdot] = (mockdf[:halo_mvir] .- mockdf[:halo_mz_1]) ./ mockdf[:halo_mz_1]
    mockdf[:Mdot][mockdf[:halo_mz_1] .== 0] = 1
    mockdf[:logMd] = log10.(mockdf[:Mdot] + 1)
    mockdf[:logc] = log10.(mockdf[:halo_nfw_conc])
    mockdf[:logspin] = log10.(mockdf[:halo_spin])
    mockdf[:logρ] = mockdf[:delta]
    mockdf[:logRe] = ones(mockdf[:log10M]) * meanr
    mockdf[:logsurf] = ones(mockdf[:log10M]) * means
    mockdf[:logv] = ones(mockdf[:log10M]) * meanv

    return mockdf
end


"""
Based on the mass range and variable configuration being considered,
this function sets the bin edges for the modeling.
"""
function set_bins(datdf, sample_num::Int, varswitch::Array{Int})

    # Define the bin edges of the galaxy properties
    if sample_num == 1
        mbins = [9.4, 9.5, 9.6, 9.7, 9.8]
    elseif sample_num == 2
        mbins = [9.8, 9.9, 10.0, 10.1, 10.2, 10.3]
    elseif sample_num == 3
        mbins = [10.3, 10.4, 10.5, 10.6]
    elseif sample_num == 4
        mbins = [10.6, 10.7, 10.8, 10.9, 11.0]
    end
    dbins = Array(linspace(-0.8, 1.0, 9))

    if varswitch[1] == 1
        rbins = [-3, -2.75, -2.5, -2.25, -2, -1.75]
    else
        rbins = [minimum(datdf[:logRe]), maximum(datdf[:logRe]) + 1]
    end

    if varswitch[2] == 1
        sbins = [8.5, 8.7, 8.9, 9.1, 9.3, 9.5]
    else
        sbins = [minimum(datdf[:logsurf]), maximum(datdf[:logsurf]) + 1]
    end

    if varswitch[3] == 1
        vbins = [1.9, 2.02, 2.14, 2.26, 2.38, 2.5]
    else
        vbins = [minimum(datdf[:logv]), maximum(datdf[:logv]) + 1]
    end

    return mbins, dbins, rbins, sbins, vbins
end


"""
This function makes arrays of dataframes separated by stellar mass and
environment density bins, as required for computing the histograms.
"""
function make_dfarrs(datdf::DataFrame, mockdf::DataFrame,
                     mbins::Array{Float64}, dbins::Array{Float64},
                     nmbins::Int, ndbins::Int)

    const datcols = [:log10M, :logρ, :logRe, :logsurf, :logv]
    const mockcols = [:logMh, :logMd, :logc, :logspin, :log10M, :logρ, :logRe, :logsurf, :logv]

    # build dataframe arrays for making galaxy counts within each of the mass-environment histogram bins
    deltadat = Array{Array{DataFrame, 1}}(nmbins)

    for i in 1:nmbins
        deltadat[i] = Array{DataFrame}(ndbins)
        massdf = datdf[mbins[i] .<= datdf[:log10M] .< mbins[i + 1], datcols]
        for j in 1:ndbins
            deltadat[i][j] = massdf[dbins[j] .<= massdf[:logρ] .< dbins[j + 1], datcols]
        end
    end

    # and similarly for the mock
    mockdfarr = Array{Array{DataFrame, 1}}(nmbins)

    for i in 1:nmbins
        mockdfarr[i] = Array{DataFrame}(ndbins)
        massdf = mockdf[mbins[i] .<= mockdf[:log10M] .< mbins[i + 1], mockcols]
        for j in 1:ndbins
            mockdfarr[i][j] = massdf[dbins[j] .<= massdf[:logρ] .< dbins[j + 1], mockcols]
        end
    end

    return deltadat, mockdfarr
end


"""
Based on the provided data sample and parameter configuration flags, this function
prepares the observations and the mock dataframes for the modeling.
"""
function prepare_dat(data_sample::Int,
                     param_config::Int)

    if !(data_sample in 1:4)
        throw(ArgumentError("data_sample must be an integer from 1-4"))
    end

    if !(param_config in 1:7)
        throw(ArgumentError("param_config must be an integer from 1-7"))
    end

    datdf, vol = get_dat(data_sample)
    meanvars = get_means(data_sample, datdf)
    mockdf = get_mock(data_sample, meanvars...)

    # create a parameter switch array from the param_config value
    varswitch = get_vars(param_config)

    # use the data sample and switch to set the bin edges
    mbins, dbins, rbins, sbins, vbins = set_bins(datdf, data_sample, varswitch)

    # store the number of mass and environment density bins
    nmbins = length(mbins) - 1
    ndbins = length(dbins) - 1

    deltadat, mockdfarr = make_dfarrs(datdf, mockdf,
                                      mbins, dbins,
                                      nmbins, ndbins)

    # get the observations
    obsdat = bincounts(deltadat, rbins, sbins, vbins)

    # create a mask of the nonzero elements of the observations
    nonzmask = find(obsdat .> 0)

    # and a tuple of the sums of the data and the log of the data
    obsums = (sum(obsdat),
              sum(obsdat[nonzmask] .* (log.(obsdat[nonzmask]))))

    # Get the conversion factor to compare the simulation volume with the data
    volfac = vol / 400^3

    return meanvars, varswitch, rbins, sbins, vbins, mockdfarr, obsdat, nonzmask, obsums, volfac
end


"""
Run an optimisation routine to find the best-fit parameters for a given model
on a given sample.
"""
function run_optim(data_sample::Int,
                   param_config::Int)

    # generate variables for optimisation
    const (meanvars, varswitch,
           rbins, sbins, vbins,
           mockdfarr,
           obsdat, nonzmask, obsums, volfac) = prepare_dat(data_sample,
                                                           param_config)
    nvars = sum(varswitch)

    # and add method to model bincount generation
    anon_bincounts = params -> gen_mod_bincounts(params, mockdfarr, varswitch,
                                                 rbins, sbins, vbins)

    # and now to set the likelihood function
    function like(params::Array{Float64,1})

        mockhist = anon_bincounts(params)
        mockhist = mockhist * volfac + exp(-125)  #

        score = cstat(mockhist, obsdat, obsums, nonzmask)

        score
    end

    # set the number of parameters and walkers from the varswitch array
    ndims = 6 * nvars
    npop = ndims * 12

    lower = -20.0 * ones(ndims)
    upper = 20.0 * ones(ndims)

    # let intercept and linear terms be larger in magnitude
    for i in 1:nvars
        lower[(6 * (i - 1) + 1):(6 * (i - 1) + 2)] = -150.0
        upper[(6 * (i - 1) + 1):(6 * (i - 1) + 2)] = 150.0
    end

    optctrl = BlackBoxOptim.bbsetup(like;  # Method = :dxnes,\n",
                                    SearchRange = collect(zip(lower, upper)),
                                    # Population = optchain,
                                    PopulationSize = npop)

    res = BlackBoxOptim.bboptimize(optctrl; MaxSteps = 30000)

    # grab the optimal value from the run to save
    best = BlackBoxOptim.best_candidate(res)

    # write to file
    datdir = get_datdir()
    th, varstring = get_thres_varstring(data_sample, varswitch)
    writedlm(string(datdir, "/optim/M", th, "/", varstring, ".dat"), best)
end


function get_chain_likevals(sample_num::Int,
                            var_config::Int,
                            chain_range::UnitRange{Int};
                            subdir = nothing)

    varswitch = get_vars(var_config)
    th, varstr = get_thres_varstring(sample_num, varswitch)

    datdir = get_datdir()

    if subdir != nothing
        dirstr = string(datdir, "/mcmc/M", th, "/", varstr, "/", subdir, "/")
    else
        dirstr = string(datdir, "/mcmc/M", th, "/", varstr, "/")
    end

    println(dirstr)

    chains = glob("*.chain", dirstr)
    likevals = glob("*.likevals", dirstr)

    nfiles = size(chains)[1]

    if length(chain_range) > nfiles
        throw(BoundsError("chain_range is not compatible with existing files"))
    end

    chainnum = length(chain_range)

    # check chain dimensions
    first = chain_range[1]
    testchain = readdlm(string(dirstr, "$first.chain"))
    ndims, nsamps = size(testchain)

    #ndims = 6 * sum(varswitch)
    #nsamps = ndims * 12 * 50

    chainarr = Array{Float64}(ndims, nsamps * chainnum)
    likearr = Array{Float64}(nsamps * chainnum)

    j = 0
    for i in chain_range

        chain = readdlm(string(dirstr, "$i.chain"))
        like = readdlm(string(dirstr, "$i.likevals"))

        chainarr[:, (nsamps * j + 1):(nsamps * (j + 1))] = chain[:, :]
        likearr[(nsamps * j + 1):(nsamps * (j + 1))] = like[:, 1]

        j += 1
    end

    return chainarr, likearr
end


"""
Function which returns the set of parameters that has the highest probability
from a set of posterior samples.
"""
function get_max_post(sample_num::Int,
                      var_config::Int,
                      chain_range::UnitRange{Int};
                      subdir = nothing)

    chain, postvals = get_chain_likevals(sample_num,
                                         var_config,
                                         chain_range,
                                         subdir = subdir)

    maxval = maximum(postvals)
    maxinds = find(postvals .== maxval)
    maxapost = chain[:, maxinds[1]]

    return maxapost
end


"""
This function reads in an existing file of posterior samples and computes the
means and variances in each of the parameters.
"""
function chain_meanvars(sample_num, var_config; subdir = nothing)

    means = zeros(6)
    vars = zeros(6)
    chain, likevals = get_chain_likevals(sample_num,
                                         var_config,
                                         75:100,
                                         subdir = subdir)

    for i in 1:6
        means[i] = mean(chain[i, :])
        vars[i] = var(chain[i, :])
    end

    return means, vars
end


"""
This function generates a set of means and variances for gaussians in each
parameter to compute the prior probabilities over the parameters. For cases
where more than one variable are being fit for, it looks for posterior chains
from single variable runs to get prior probabilities. Otherwise the mean value
for the observations of the variable are assumed for the intercept of the
model, with zero mean for the other parameters and wide variances to remain
conservative while penalising large parameter values.
"""
function get_prior_means_invvars(sample_num,
                                 varswitch,
                                 obsmeans;
                                 subdir = nothing)

    nvars = sum(varswitch)  # number of data variables

    datdir = get_datdir()

    # initialise return arrays
    means = zeros(6 * nvars)
    invvars = 0.04 * ones(6 * nvars)  # base assume a stddev of 5

    if nvars > 1

        j = 1
        for (i, v) in enumerate(varswitch)

            if v == 1

                # grab means and variances of single variable posterior
                chainmvs = chain_meanvars(sample_num,
                                          2^(3 - i),
                                          subdir = subdir)
                means[(6 * (j - 1) + 1):(6 * j)] = chainmvs[1]
                invvars[(6 * (j - 1) + 1):(6 * j)] = 1 ./ chainmvs[2]

                # increase value of j for each variable found
                j += 1
            end
        end

    else
        invvars[1:3] = 0.001  # mass parameters are not penalised much
    end

    return means, invvars
end


"""
This function computes a prior probability for a set of parameters given a set
of gaussian means and variances in each parameter.
"""
function logprior(params, means_invvars)

    means, invvars = means_invvars

    logfac = -0.5 * log(2π * 25.0^(length(params)))

    return logfac - 0.5 * sum(invvars .* ((params .- means) .^ 2))
end


"""
Run MCMC using AffineInvariantMCMC for a given data sample and parameter space.
Uses the optimised values for the parameters to generate walker initialisation.
"""
function run_mcmc(data_sample::Int,
                  param_config::Int;
                  subdir = nothing,
                  outdir = nothing)

    # generate variables again for posterior sampling
    const (meanvars, varswitch,
           rbins, sbins, vbins,
           mockdfarr,
           obsdat, nonzmask, obsums, volfac) = prepare_dat(data_sample,
                                                           param_config)
    nvars = sum(varswitch)

    # set data input/output variables
    datdir = get_datdir()
    th, varstring = get_thres_varstring(data_sample, varswitch)

    # and add the right method to model bincounts
    anon_bincounts = params -> gen_mod_bincounts(params, mockdfarr, varswitch,
                                                 rbins, sbins, vbins)

    # add a method for the prior
    prior_mvs = get_prior_means_invvars(data_sample,
                                        varswitch,
                                        meanvars,
                                        subdir = subdir)
    anon_prior = params -> logprior(params, prior_mvs)

    # and now to set the logposterior for the sampler
    function lnprob(params::Array{Float64,1})

        lnprior = anon_prior(params)

        mockhist = anon_bincounts(params)
        mockhist = mockhist * volfac + exp(-350)  #

        if sum(mockhist) == 0
            return -Inf
        end

        score = cstat(mockhist, obsdat, obsums, nonzmask)

        return lnprior - score
    end

    # set parameters for the sampler
    numdims = 6 * nvars
    numwalkers = numdims * 20
    thinning = 5
    numsamples_perwalker = 50
    burnin = 10

    ### This is just optional code for testing alternative initialisations
    # check if an alternative (non-optimisation) set of parameters exists
    if isfile(string(datdir, "/mcmc/M", th, "/", varstring, ".alt"))
        pars = readdlm(string(datdir, "/mcmc/M", th, "/", varstring, ".alt"))
        pars = squeeze(pars, 2)
        par_stdevs = 0.01
        println("alt initialisation is running, parameters are:")
        println(pars)
    else

        # otherwise, if a previous run exists, grab maximum a posteriori
        if nvars == 1 && subdir != nothing

            pars = get_max_post(data_sample,
                                param_config,
                                75:100,
                                subdir = subdir)
            par_stdevs = 0.01
            println("running single variable mcmc with initial parameters:")
            println(pars)

        # get max posterior values for all sub variables if fitting multiple
        elseif nvars > 1 && subdir != nothing

            pars = zeros(numdims)

            j = 1
            for (i, v) in enumerate(varswitch)

                if v == 1

                    chainmax = get_max_post(data_sample,
                                            2^(3 - i),
                                            75:100,
                                            subdir = subdir)
                    pars[(6 * (j - 1) + 1):(6 * j)] = chainmax

                    j += 1
                end
            end

            par_stdevs = 0.01

            println("got the max posterior vals for all parameters:")
            println(pars)

        # alternatively, use the prior means
        elseif prior_mvs[1][2] != 0
            pars = prior_mvs[1]
            par_stdevs = prior_mvs[2]

        # otherwise just use the optimisation values
        else
            # read best fit optimisation parameters
            pars = readdlm(string(datdir, "/optim/M", th, "/", varstring, ".dat"))
            pars = squeeze(pars, 2)
            par_stdevs = 0.05
        end
    end

    # initialise walkers
    x0 = par_stdevs .* randn(numdims, numwalkers)
    x0 = x0 .+ pars

#     println("lnprobs for the initial samples:")
#     for i in 1:numwalkers
#         println(lnprob(x0[:, i]))
#     end

#     println("burning in...")
    chain, likevals = AffineInvariantMCMC.sample(lnprob, numwalkers, x0, burnin, 1)
#     println("lnprobs after burnin:")
#     println(likevals)

    println("and starting mcmc...")
    for i in 1:100
        chain, likevals = AffineInvariantMCMC.sample(lnprob, numwalkers, chain[:, :, end], numsamples_perwalker, 1)
#         println("lnprobs after one round of mcmc:")
#         println(likevals)
#         error("good nuff")
        flatchain, flatlikevals = AffineInvariantMCMC.flattenmcmcarray(chain, likevals)

        if outdir != nothing
            outpath = string(datdir, "/mcmc/M", th, "/", varstring, "/", outdir)
        else
            outpath = string(datdir, "/mcmc/M", th, "/", varstring)
        end

        writedlm(string(outpath, "/$i.chain"), flatchain)
        writedlm(string(outpath, "/$i.likevals"), flatlikevals)
    end
end


if length(ARGS) > 0 && length(ARGS) >= 2

    # first argument passed is number sample to run mcmc for
    num_sample = parse(Int, ARGS[1])

    # second argument passed dictates which parameter configuration to use
    param_val = parse(Int, ARGS[2])

    # optional third argument sets chain subdirectory name
    # for previous samples
    if length(ARGS) >= 3
        subdir = ARGS[3]
        println("using subdir: ", subdir)
    else
        subdir = nothing
    end

    if length(ARGS) >= 4
        outdir = ARGS[4]
        println("using outdir: ", outdir)
    else
        outdir = nothing
    end


    # check if parameters have been optimised
    datdir = get_datdir()
    th, varstring = get_thres_varstring(num_sample, param_val)
    bestparams = string(datdir, "/optim/M", th, "/", varstring, ".dat")

    # optimise the parameters for a first guess if necessary
    if isfile(bestparams)
        println("# # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        println("best-fit parameter guess exists, skipping optimisation")
        println("# # # # # # # # # # # # # # # # # # # # # # # # # # # #")
    else
        println("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        println("First, to find a guess for the best-fit parameters via optimisation")
        println("# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #")
        run_optim(num_sample, param_val)
    end

    # and now get an mcmc chain
    println("# # # # # # # # # # # # # # # # # # # # # # # # # # #")
    println("now running mcmc initialised near the best-fit guess")
    println("# # # # # # # # # # # # # # # # # # # # # # # # # # #")
    run_mcmc(num_sample, param_val, subdir = subdir, outdir = outdir)

elseif length(ARGS) > 0 && length(ARGS) != 2
    println("error: incorrect number of arguments")
    println("usage: julia run_mcmc.jl <sample number> <parameter config>")
end
