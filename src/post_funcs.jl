include("./run_mcmc.jl")
include("./obs_funcs.jl")
include("./fit_funcs.jl")
include("./Jackknife.jl")

using Glob


function get_chain_likevals(sample_num::Int,
                            var_config::Int,
                            chain_range::UnitRange{Int})

    varswitch = get_vars(var_config)
    th, varstr = get_thres_varstring(sample_num, varswitch)
    dirstr = string("../dat/mcmc/M", th, "/", varstr, "/")

    chains = glob(string(dirstr, "*.chain"))
    likevals = glob(string(dirstr, "*.likevals"))

    nfiles = size(chains)[1]

    if chain_range[end] > nfiles
        throw(BoundsError("chain_range is not compatible with existing files"))
    end

    chainnum = length(chain_range)
    ndims = 6 * sum(varswitch)
    nsamps = ndims * 12 * 50
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


function post_draw_ribbons(sample_num::Int,
                           var_config::Int;
                           dbinedges::Array{Float64} = Array(linspace(-0.8, 1, 9)),
                           yticks = nothing,
                           ndraws::Int = 50,
                           chain_range::UnitRange{Int} = 75:100)

    # set up bin values
    ndbins = length(dbinedges) - 1
    minmass = [9.4, 9.8, 10.3, 10.6][sample_num]
    maxmass = [9.8, 10.3, 10.6, 11.0][sample_num]

    # set up variable names
    varswitch = get_vars(var_config)
    varnames = ["Rₑ", "ρₛ", "σᵥ"]
    datcols = [:R_e, :surfdensR_eo2, :vdisp][Bool.(varswitch)]
    mockcols = [:logRe, :logsurf, :logv][Bool.(varswitch)]
    nvars = sum(varswitch)
    cols = varnames[Bool.(varswitch)]
    cencols = [Symbol(string("cen", col)) for col in cols]

    # create a plot object and an array to hold plot layers for each variable
    subplots = Array{Gadfly.Plot}(nvars, 1)
    sublayers = [[] for i in 1:nvars]

    # get the data
    datdf, vol = get_dat(sample_num)
    datdf = datdf[minmass .<= datdf[:log10M] .< maxmass, :]

    # get the mock
    means = get_means(sample_num, datdf)
    mockdf = get_mock(sample_num, means...)
    mockdf = mockdf[minmass .<= mockdf[:log10M] .< maxmass, :]

    # find the mean densities
    datdmeans = zeros(ndbins)
    mockdmeans = zeros(ndbins)
    for j in 1:ndbins
        datdmeans[j] = mean(datdf[:logρ][dbinedges[j] .<= datdf[:logρ] .< dbinedges[j + 1]])
        mockdmeans[j] = mean(mockdf[:logρ][dbinedges[j] .<= mockdf[:logρ] .< dbinedges[j + 1]])
    end

    # and get the plot going
    xlabel = " log₁₀(δ + 1)"
    ylabels = []

    # loop over the variables for the data
    for k in 1:nvars

        # start with computing data values
        col = cols[k]
        datdf[Symbol(col)] = datdf[datcols[k]]
        append!(ylabels, "Δ$col / ⟨$col⟩ₘ")
#        yticks =

        # control for mass variations within the bin
        cencol = cencols[k]
        datdf[cencol] = rm_mean_logmass(datdf[Symbol(col)], datdf[:log10M])
        datdf[cencol] = datdf[cencol] / mean(datdf[Symbol(col)])

        valmeans, valcovars = jackmeansvars(envmeans,
                                            ndbins,
                                            datdf,
                                            Array(datdf[:jackvol]),
                                            (cencol, dbinedges,))
        valerrs = sqrt.(diag(valcovars))
        valmins = valmeans .- valerrs
        valmaxs = valmeans .+ valerrs

        append!(sublayers[k], layer(x=datdmeans,
                                    y=valmeans,
                                    Geom.point,
                                    Theme(point_size=1.0mm,
                                    discrete_highlight_color=(u -> LCHab(0,0,0)))))

        append!(sublayers[k], layer(x=datdmeans,
                                    y=valmeans,
                                    ymin=valmins,
                                    ymax=valmaxs,
                                    color=[colorant"black"],
                                    Geom.errorbar))
    end

    # and now to get the mcmc draws for the mock
    chain, likevals = get_chain_likevals(sample_num,
                                         var_config,
                                         chain_range)
    for i in 1:ndraws

        # draw random set of parameters from posterior and generate values
        pars = chain[:, rand(1:end)]
        gen_obs([[mockdf]], pars)

        # make plot for each variable
        for k in 1:nvars

            col = cols[k]
            cencol = cencols[k]
            mockdf[Symbol(col)] = 10 .^ mockdf[mockcols[k]]

            mockdf[cencol] = rm_mean_logmass(mockdf[Symbol(col)], mockdf[:log10M])
            mockdf[cencol] = mockdf[cencol] / mean(mockdf[Symbol(col)])

            mlemeans, mlcovars = jackmeansvars(envmeans,
                                               ndbins,
                                               mockdf,
                                               Int.(mockdf[:subvol]),
                                               (cencol, dbinedges,))
            mlerrs = sqrt.(diag(mlcovars))
            mlemins = mlemeans .- mlerrs
            mlemaxs = mlemeans .+ mlerrs

            append!(sublayers[k], layer(x=mockdmeans,
                                        y=mlemeans,
                                        ymin=mlemins,
                                        ymax=mlemaxs,
                                        Geom.line,
                                        Geom.ribbon,
                                        Theme(lowlight_color=c->RGBA{Float32}(c.r, c.g, c.b, 0.1))))
        end
    end


    for k in 1:nvars

        # if yticks of variables were given, add to plot arguments
        if yticks != nothing
            append!(sublayers[k], Guide.yticks(ticks=yticks[k]))
        end

        subplots[k] = plot(sublayers[k]...,
                           Guide.xlabel(xlabel),
                           Guide.ylabel(ylabels[k]))
    end

    gridstack(subplots)
end


function posterior_jackknife(sample_num::Int,
                             dbinedges::Array{Float64} = Array(linspace(-0.8, 1, 9)),
                             ndraws::Int = 50)


    # loop over the variables
    for k in 1:nvars

        # start with computing data values
        col = varnames[Bool.(varswitch)][k]
        datdf[Symbol(col)] = datdf[datcols[k]]
#        yticks =

        # control for mass variations within the bin
        cencol = Symbol(string("cen", col))
        datdf[cencol] = rm_mean_logmass(datdf[Symbol(col)], datdf[:log10M])
        datdf[cencol] = datdf[cencol] / mean(datdf[Symbol(col)])

        valmeans, valcovars = jackmeansvars(envmeans,
                                            ndbins,
                                            datdf,
                                            Array(datdf[:jackvol]),
                                            (cencol, dbinedges,))
        valerrs = sqrt.(diag(valcovars))
        valmins = valmeans .- valerrs
        valmaxs = valmeans .+ valerrs

        # and now for the mock

        subplots[k] = plot(layer(x=datdmeans,
                                 y=valmeans,
                                 Geom.point,
                                 Theme(point_size=1.0mm,
                                       discrete_highlight_color=(u -> LCHab(0,0,0)))),
                           layer(x=datdmeans,
                                 y=valmeans,
                                 ymin=valmins,
                                 ymax=valmaxs,
                                 color=[colorant"black"],
                                 Geom.errorbar),
                           layer(x=mockdmeans,
                                 y=mlemeans,
                                 ymin=mlemins,
                                 ymax=mlemaxs,
                                 Geom.line,
                                 Geom.ribbon),
                           Guide.yticks(ticks=yticks),
                           Guide.xlabel(xlabel),
                           Guide.ylabel(ylabel))

    end




end
