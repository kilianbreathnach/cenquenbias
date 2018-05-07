using DataFrames
using Cosmology
using Gadfly
using Compose
include("./Jackknife.jl")
using .Jackknife


"""
This function loads the catalog from papers I, II, and III. It returns a
set of galaxies that are found to be centrals with either 50% probability
or 99% probability. It can return either quenched, or star-forming centrals.
"""
function load_groupdat(; fullpure = "pure", quenform = "quen")

    # get the data directory to pull the galaxies from
    datdir = join(split(Base.@__FILE__, "/")[1:(end - 2)], "/")
    datdir = string(datdir, "/dat")

    fulldf = DataFrame()

    # these are the column names for the dataframe
    probnames = [:foo, :galid, :groupid, :cenid, :rband, :Psat, :Mh, :foo2, :foo3, :foo4, :projR, :projrad, :angRh]
    corrnames = [:foo5, :galid, :M_r, :M_g, :cz, :Dn4000, :H_delta, :logsSFR, :stelM, :ra, :dec, :vdisp, :S2N, :sersic,
                 :conc, :KplusA, :R_exp, :surfdens1kpc, :surfdensR_e, :surfdensR_eo2, :vdisp_err, :Bulge2Tlr, :B2T_err,
                 :GMoR_e, :R_e]

    probfiles = [string(datdir, "/groups/clf_groups_M18_M9.4.prob"),
                 string(datdir, "/groups/clf_groups_M19_M9.8.prob"),
                 string(datdir, "/groups/clf_groups_M20_M10.3.prob")]
    corrfiles = [string(datdir, "/groups/clf_groups_M18_M9.4.galdata_corr"),
                 string(datdir, "/groups/clf_groups_M19_M9.8.galdata_corr"),
                 string(datdir, "/groups/clf_groups_M20_M10.2.galdata_corr")]
    densfiles = [string(datdir, "/groups/densities/density_r10.M18_M9.4"),
                 string(datdir, "/groups/densities/density_r10.M19_M9.8"),
                 string(datdir, "/groups/densities/density_r10.M20_M10.3")]
    randfiles = [string(datdir, "/groups/densities/drandom_r10.M18_M9.4"),
                 string(datdir, "/groups/densities/drandom_r10.M19_M9.8"),
                 string(datdir, "/groups/densities/drandom_r10.M20_M10.3")]

    # store max redshift for computing sample volumes
    zmaxs = zeros(3)
    zmins = zeros(3)

    for i in 1:3

        # read the two catalogues in and join them on galaxy id
        probdf = readtable(probfiles[i], separator=' ', header=false)
        names!(probdf, probnames)
        corrdf = readtable(corrfiles[i], separator=' ', header=false)
        names!(corrdf, corrnames)

        # add a column for the density, corrected with the randoms
        density = readdlm(densfiles[i])
        rands = readdlm(randfiles[i])
        zi = find(rands[:, 1] .== 0)
        ρ_corr = 1.25 * (density[:, 1] ./ rands[:, 1])
        probdf[:ρ_env] = DataArray(ρ_corr)

        # remove nan value (there was one row in rands that had a zero)
        probdf = probdf[.!isnan.(probdf[:ρ_env]), :]

        joindf = join(probdf, corrdf, on=:galid)

        # take max z val
        zmaxs[i] = maximum(joindf[:cz])
        zmins[i] = minimum(joindf[:cz])

        if i == 1
            fulldf = joindf
        else
            fulldf = [fulldf; joindf]
        end
    end

    fulldf = unique(fulldf, :galid)

    # remove any galaxies in environment density less than or equal to zero
    # to avoid computational issues
    fulldf = fulldf[fulldf[:ρ_env] .> 0.0, :]

    # remove bad measurements (i.e. where they are zero) and outliers
    fulldf = fulldf[fulldf[:R_e] .> 0.0, :]
    fulldf = fulldf[fulldf[:vdisp] .> 0.0, :]
    fulldf = fulldf[fulldf[:surfdensR_eo2] .> 0, :]
    fulldf = fulldf[fulldf[:Dn4000] .< 2.5, :]
    fulldf = fulldf[fulldf[:Dn4000] .> 1.0, :]
    fulldf = fulldf[fulldf[:vdisp] .< 500, :]
    fulldf = fulldf[fulldf[:R_e] .< 0.05, :]

    # take log of R_e and add some other convenient columns
    fulldf[:logMh] = log10.(fulldf[:Mh])
    fulldf[:Dₙ4000] = fulldf[:Dn4000]
    fulldf[:logRe] = log10.(fulldf[:R_e])
    fulldf[:logsurf] = log10.(fulldf[:surfdensR_eo2])

    # get data deltas
    fulldf[:delta] = fulldf[:ρ_env] - 1

    # add a column for log10 of stellar mass and log of environment
    fulldf[:log10M] = log10.(fulldf[:stelM])
    fulldf[:logρ] = log10.(fulldf[:ρ_env])

    # Now take all the rows out which don't have central galaxies
    if fullpure == "full"
        fulldf = fulldf[fulldf[:Psat] .< 0.5, :]
    elseif fullpure == "pure"
        fulldf = fulldf[fulldf[:Psat] .< 0.01, :]
    end

    # Take only the quenched galaxies
    if quenform == "quen"
        fulldf = fulldf[fulldf[:Dn4000] .> 1.6, :]
    elseif quenform == "form"
        fulldf = fulldf[fulldf[:Dn4000] .< 1.6, :]
    end

    # compute the comoving volume info
    zmin = zmins[1] / 299792
    zmaxs = zmaxs / 299792

    # set a cosmology
    cosmo = cosmology(h = 1.0, OmegaM = 0.286)

    minvol = comoving_volume_gpc3(cosmo, zmin) * 1000^3
    vols = [comoving_volume_gpc3(cosmo, z) for z in zmaxs] * 1000^3 - minvol
    vols = 0.18952 * vols  # fraction of sky from combmask

    return fulldf, vols
end


"""
This function removes the trend in galaxy logmass from another galaxy variable.
"""
function rm_mean_logmass(vararr, logmass; nbins = 10)

    maxM = maximum(logmass)
    minM = minimum(logmass)
    Medges = linspace(minM, maxM, nbins + 1)

    cenvar = zeros(length(vararr))

    for i in 1:nbins
        inds = Medges[i] .<= logmass .<= Medges[i + 1]
        meanvar = mean(vararr[inds])
        cenvar[inds] = vararr[inds] - meanvar
    end

    return cenvar
end


"""
This function computes the mean value of a galaxy property within a set of
bins in environment density.
"""
function envmeans(subdf::DataFrames.DataFrame,
                  cencol::Symbol, dbinedges::Array{Float64, 1})
    ndbins = length(dbinedges) - 1
    ems = Array{Float64}(ndbins)
    for j in 1:ndbins
        ems[j] = mean(subdf[cencol][dbinedges[j] .<= subdf[:logρ] .<= dbinedges[j + 1]])
    end
    ems
end


"""
This function produces one of the paper plots, showing the mass-controlled
variation in galaxy properties as a function of environment density within
specified mass ranges.
"""
function massenv_plot(galdf, col,
                      yticksarr, annotarr,
                      dbinedges, dmeans,
                      samplevols,
                      logMedges = [9.4, 9.8, 10.3, 10.6, 11.0];
                      ptcolor = "blue",
                      nsubvols = 25)

    nmbins = length(logMedges) - 1
    subplots = Array{Gadfly.Plot, 1}(nmbins)

    for i in 1:nmbins

        # get the samples from the mass bin for each subplot
        subdf = galdf[logMedges[i] .<= galdf[:log10M] .< logMedges[i + 1], :]

        # control for mass variations within the bin
        cencol = Symbol(string("cen", col))
        subdf[cencol] = rm_mean_logmass(subdf[Symbol(col)], subdf[:log10M])
        subdf[cencol] = subdf[cencol] / mean(subdf[Symbol(col)])

        # now get the values for each bin in density, computing the errors
        # using jackknife
        jackvol = samplevols[i] / nsubvols
        valmeans, valcovars = jackmeansvars(envmeans, length(ndbins),
                                            subdf, Array(subdf[:jackvol]),
                                            (cencol, dbinedges,))
        valerrs = sqrt.(diag(valcovars))
        valmins = valmeans .- valerrs
        valmaxs = valmeans .+ valerrs

        # get axis labels and ticks for panel plots
        xlabel = " log₁₀(δ + 1)"

        if i == 1
            ylabel = "Δ$col / ⟨$col⟩<sub>M</sub>"
        else
            ylabel = nothing
        end
        yrange = yticksarr[i]  # need different ranges for each plot
        anvals = annotarr[i]

        # compute spearman rank correlation
        rₛ = cor(sort(subdf[cencol]), sort(subdf[:log10M]))

        # and compute chi-square of zero line
        chisquare = sum((valmeans .^ 2) ./ (valerrs .^ 2))
        redchi = chisquare / (ndbins - 2)
        pval = ccdf(Chisq(ndbins - 2), chisquare)

        # print these values, just to be clear
        println(Formatting.format("logMₛ = {1:.1f}", mlabels[i]))
        println(Formatting.format("rₛ = {1:.2f}", rₛ))
        println(Formatting.format("χ² = {1:.1f}", chisquare))
        println(Formatting.format("χ²_ν = {1:.1f}", redchi))
        println(Formatting.format("p = {1:.3f}", pval))
        println(valcovars)
        println("   ")

        plotlayers = [layer(x=dmeans[i, :], y=valmeans, Geom.point,
                            Theme(point_size=1.0mm,
                                  default_color=ptcolor,
                                  discrete_highlight_color=(u -> LCHab(0,0,0)))),
                      layer(x=dmeans[i, :], y=valmeans, ymin=valmins, ymax=valmaxs,
                            color=[colorant"black"], Geom.errorbar)]

        subplots[i] = plot(plotlayers...,
                           Guide.yticks(ticks=yrange),
                           Guide.xlabel(xlabel), Guide.ylabel(ylabel),
                           Guide.annotation(compose(context(),
                                                    Compose.text(anvals[1], yrange[anvals[2]],
                                                                 Formatting.format("log₁₀(M<sub>*</sub>) = {1:.1f}", mlabels[i])))),
                           Guide.annotation(compose(context(),
                                                    Compose.text(anvals[1], yrange[anvals[2] - 1],
                                                                 Formatting.format("χ²<sub>red</sub> = {1:.1f}", redchi)))))
    end

    fig = hstack(subplots[1:4])

    return fig
end
