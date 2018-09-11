include("./run_mcmc.jl")


function post_draw_ribbons(sample_num::Int,
                           var_config::Int;
                           dbinedges::Array{Float64} = Array(linspace(-0.8, 1, 9)),
                           yticks = nothing,
                           ndraws::Int = 50,
                           chain_range::UnitRange{Int} = 75:100,
                           like_cutoff::Float64 = -Inf,
                           subdir = nothing)

    # set up bin values
    ndbins = length(dbinedges) - 1
    minmass = [9.4, 9.8, 10.3, 10.6][sample_num]
    maxmass = [9.8, 10.3, 10.6, 11.0][sample_num]

    # set up variable names
    varswitch = get_vars(var_config)
    varnames = ["Rₑ", "ρₛ", "σᵥ"]
    varcolors = [colorant"orange", colorant"magenta", colorant"green"]
    datcols = [:R_e, :surfdensR_eo2, :vdisp][Bool.(varswitch)]
    mockcols = [:logRe, :logsurf, :logv][Bool.(varswitch)]
    nvars = sum(varswitch)
    cols = varnames[Bool.(varswitch)]
    ptcolors = varcolors[Bool.(varswitch)]
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
        push!(ylabels, string("Δ$col / ⟨$col⟩ₘ"))
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
                                          discrete_highlight_color=(u -> LCHab(0,0,0)),
                                          default_color = ptcolors[k])))

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
                                         chain_range,
                                         subdir = subdir)
    chain = chain[:, likevals .> like_cutoff]

    for i in 1:ndraws

        # draw random set of parameters from posterior and generate values
        pars = chain[:, rand(1:end)]
        gen_obs(pars, [[mockdf]], varswitch)

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
                                               (cencol, dbinedges,),
                                               nvols = 27)
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


# function all_param_plot(;
#                         subdir = "max_init",
#                         chain_range::UnitRange{Int} = 75:100)
#
#     subplots = Array{Gadfly.Plot}(3, 4)
#
#     # some arrays for plot names and colours and yranges
#     massrange = ["[9.4, 9.8]", "[9.8, 10.3]", "[10.3, 10.6]", "[10.6, 11.0]"]
#     coeffnames = ["β<sub>Ṁ</sub>", "β<sub>c</sub>", "β<sub>λ</sub>"]
#
#     colors = [convert(LCHuv, colorant"orange"),
#               convert(LCHuv, colorant"magenta"),
#               convert(LCHuv, colorant"green"),
#               convert(LCHuv, colorant"black")]
#
#     ytickarr = [Array(linspace(-5, 3, 9)),
#                 Array(linspace(-2, 2, 5)),
#                 Array(linspace(-3, 2, 6))]
#
#     # loop over mass bins
#     for (i, th) in enumerate(["18", "19", "20a", "20b"])
#
#         # and matrices for the variable-coefficient means and errs
#         cmeans = zeros(3, 3)
#         cmins = zeros(3, 3)
#         cmaxs = zeros(3, 3)
#
#         # loop over variables
#         for (j, varname) in enumerate(["Re", "surf", "vdisp"])
#
#             chain, likevals = get_chain_likevals(i, 2^(3 - j),
#                                                  chain_range,
#                                                  subdir = subdir)
#
#             # loop over parameter coefficients
#             for k in 1:3
#
#                 cmean = mean(chain[3 + k, :])
#                 cvar = var(chain[3 + k, :])
#                 cerr = sqrt(cvar)
#
#                 cmeans[j, k] = cmean
#                 cmins[j, k] = cmean - cerr
#                 cmaxs[j, k] = cmean + cerr
#
#             end
#         end
#
#         # now draw the column of subplots
#         for k in 1:3
#
#             if k == 3
#                 xlabel = string("log<sub>10</sub> m<sub>*</sub> = ",
#                                 massrange[i])
#             else
#                 xlabel = nothing
#             end
#
#             # 1st column has ylabels
#             if i == 1
#                 ylabel = coeffnames[k]
#             else
#                 ylabel = nothing
#             end
#
#             plotlayers = [ #coeff_layer,
#                           layer(x=[1.0, 2.0, 3.0],
#                                 color=[1, 2, 3],
#                                 y=cmeans[:, k],
#                                 shape=[Shape.xcross,
#                                        Shape.star1,
#                                        Shape.dtriangle],
#                                 Geom.point,
#                                 Theme(point_size=3mm)),
#                           layer(x=[1.0, 2.0, 3.0],
#                                 y=cmeans[:, k],
#                                 ymin=cmins[:, k],
#                                 ymax=cmaxs[:, k],
#                                 color=[4],
#                                 Geom.errorbar)]
#
#             subplots[k, i] = plot(plotlayers...,
#                                   Guide.xticks(ticks=nothing),
#                                   Guide.xlabel(xlabel), Guide.ylabel(ylabel),
#                                   Theme(key_position = :none),
#                                   Scale.color_discrete_manual(colors...))
#         end
#     end
#
#     gridstack(subplots)
# end


function get_param_means_errs(sample_num::Int,
                              var_config::Int,
                              chain_range::UnitRange{Int};
                              subdir = nothing,
                              baseind = 3,
                              cutoff = -Inf)

    chain, likevals = get_chain_likevals(sample_num,
                                         var_config,
                                         chain_range,
                                         subdir = subdir)

    mask = likevals .> cutoff

    cmeans = zeros(3)
    cmins = zeros(3)
    cmaxs = zeros(3)

    # loop over parameter coefficients
    for k in 1:3

        cmean = mean(chain[baseind + k, mask])
        cvar = var(chain[baseind + k, mask])
        cerr = sqrt(cvar)

        cmeans[k] = cmean
        cmins[k] = cmean - cerr
        cmaxs[k] = cmean + cerr

    end

    return cmeans, cmins, cmaxs
end



"""
var_num = [1, 2, 3] means [R_e, surf, vdisp] respectively
"""
function all_param_plot(var_num;
                        subdir = "max_init",
                        chain_range::UnitRange{Int} = 75:100)

    # each row is for one parameter coefficient
    # each column is for a set of mcmc chains
    subplots = Array{Gadfly.Plot}(3, 3)

    # some arrays for plot names and colours and yranges
    varnames = ["Re", "surf", "vdisp"]
    massrange = ["[9.4, 9.8]", "[9.8, 10.3]", "[10.3, 10.6]", "[10.6, 11.0]"]
    coeffnames = ["β<sub>Ṁ</sub>", "β<sub>c</sub>", "β<sub>λ</sub>"]

    varcolors = [colorant"orange", colorant"magenta", colorant"green"]
    midcolors = []  # to store the colors needed for middle column

    ytickarr = [Array(linspace(-5, 3, 9)),
                Array(linspace(-2, 2, 5)),
                Array(linspace(-3, 2, 6))]

    # cutoff values for likelihoods in each sample
    cutoffarr = [-3e5, -6e6, -1e7, -7e6]

    # loop over subplot columns
    for i in 1:3

        # set up matrices for the variable-coefficient means and errs
        if i != 2
            cmeans = zeros(4, 3)
            cmins = zeros(4, 3)
            cmaxs = zeros(4, 3)

        # need to get two sets of values for the double-variable runs
        else
            cmeans = zeros(2, 4, 3)
            cmins = zeros(2, 4, 3)
            cmaxs = zeros(2, 4, 3)
        end

        # loop over mass samples
        for (j, th) in enumerate(["18", "19", "20a", "20b"])

            if i != 2

                if i == 1
                    var_config = 2^(3 - var_num)
                    chaynrayng = chain_range
                    baseind = 3
                elseif i == 3
                    var_config = 7
                    chaynrayng = 20:42
                    baseind = 3 + 6 * (var_num - 1)
                end

                (cmeans[j, :],
                 cmins[j, :],
                 cmaxs[j, :]) = get_param_means_errs(j,
                                                     var_config,
                                                     chaynrayng,
                                                     subdir = subdir,
                                                     cutoff = cutoffarr[j],
                                                     baseind = baseind)
            else

                varinds = [1, 2, 3]
                deleteat!(varinds, var_num)

                for k in 1:2

                    # figure out index of other variable and store its colour
                    otherind = varinds[k]
                    push!(midcolors, varcolors[otherind])

                    var_config = 2^(3 - var_num) + 2^(3 - otherind)
                    if otherind < var_num
                        baseind = 9
                    else
                        baseind = 3
                    end

                    (cmeans[k, j, :],
                     cmins[k, j, :],
                     cmaxs[k, j, :]) = get_param_means_errs(j,
                                                            var_config,
                                                            chain_range,
                                                            cutoff = cutoffarr[j],
                                                            subdir = subdir)
                end

            end
        end

        # now draw the column of subplots
        for k in 1:3

            if k == 3
                xlabel = string("log<sub>10</sub> m<sub>*</sub>")
            else
                xlabel = nothing
            end

            # 1st column has ylabels
            if i == 1
                ylabel = coeffnames[k]
            else
                ylabel = nothing
            end

            if i != 2
                plotlayers = [layer(x=[9.6, 10.05, 10.45, 10.8],
                                    y=cmeans[:, k],
                                    Geom.point,
                                    Theme(point_size=2mm,
                                          discrete_highlight_color=(u -> LCHab(0,0,0)),
                                          default_color=varcolors[var_num])),
                              layer(x=[9.6, 10.05, 10.45, 10.8],
                                    y=cmeans[:, k],
                                    ymin=cmins[:, k],
                                    ymax=cmaxs[:, k],
                                    color=[colorant"black"],
                                    Geom.errorbar)]
            else
                 plotlayers = []
                 for z in 1:2
                     dx = (-1)^z * 0.05
                     append!(plotlayers, [layer(x=[9.6, 10.05, 10.45, 10.8] + dx,
                                          y=cmeans[z, :, k],
                                          Geom.point,
                                          Theme(point_size=2mm,
                                                discrete_highlight_color=(u -> LCHab(0,0,0)),
                                                default_color=midcolors[z])),
                                          layer(x=[9.6, 10.05, 10.45, 10.8] + dx,
                                                y=cmeans[z, :, k],
                                                ymin=cmins[z, :, k],
                                                ymax=cmaxs[z, :, k],
                                                color=[colorant"black"],
                                                Geom.errorbar)])
                end

            end

            xticks = [9.4, 9.8, 10.3, 10.6, 11.0]
            subplots[k, i] = plot(plotlayers...,
                                  Guide.xticks(ticks=xticks),
                                  Guide.xlabel(xlabel), Guide.ylabel(ylabel),
                                  Theme(key_position = :none))
        end
    end

    gridstack(subplots)
end


function all_param_violins(var_num;
                           subdir = "max_init",
                           chain_range::UnitRange{Int} = 75:100)

    # each row is for one parameter coefficient
    # each column is for a set of mcmc chains
    subplots = Array{Gadfly.Plot}(3, 3)

    # some arrays for plot names and colours and yranges
    varnames = ["Re", "surf", "vdisp"]
    massrange = ["[9.4, 9.8]", "[9.8, 10.3]", "[10.3, 10.6]", "[10.6, 11.0]"]
    coeffnames = ["β<sub>Ṁ</sub>", "β<sub>c</sub>", "β<sub>λ</sub>"]

    varcolors = [colorant"orange", colorant"magenta", colorant"green"]
    midcolors = []  # to store the colors needed for middle column

    ytickarr = [Array(linspace(-5, 3, 9)),
                Array(linspace(-2, 2, 5)),
                Array(linspace(-3, 2, 6))]

    # cutoff values for likelihoods in each sample
    cutoffarr = [-3e5, -6e6, -1e7, -7e6]

    # loop over subplot columns
    for i in 1:3

        # set up matrices for the variable-coefficient means and errs
        if i != 2
            csamps = Array{Array{Float64}}(4, 3)

        # need to get two sets of values for the double-variable runs
        else
            csamps = Array{Array{Float64}}(2, 4, 3)
        end

        # loop over mass samples
        for (j, th) in enumerate(["18", "19", "20a", "20b"])

            if i != 2

                if i == 1
                    var_config = 2^(3 - var_num)
                    chaynrayng = chain_range
                    baseind = 3
                elseif i == 3
                    var_config = 7
                    chaynrayng = 20:42
                    baseind = 3 + 6 * (var_num - 1)
                end

                chain, likevals = get_chain_likevals(j,
                                                     var_config,
                                                     chaynrayng,
                                                     subdir = subdir)
                chain = chain[:, likevals .> cutoffarr[j]]

                for z in 1:3
                    csamps[j, z] = chain[baseind + z, :]
                end

            else

                varinds = [1, 2, 3]
                deleteat!(varinds, var_num)

                for k in 1:2

                    # figure out index of other variable and store its colour
                    otherind = varinds[k]
                    push!(midcolors, varcolors[otherind])

                    var_config = 2^(3 - var_num) + 2^(3 - otherind)
                    if otherind < var_num
                        baseind = 9
                    else
                        baseind = 3
                    end

                    chain, likevals = get_chain_likevals(j,
                                                         var_config,
                                                         chain_range,
                                                         subdir = subdir)
                    chain = chain[:, likevals .> cutoffarr[j]]

                    for z in 1:3
                        csamps[k, j, z] = chain[baseind + z, :]
                    end

                end

            end
        end

        # now draw the column of subplots
        for k in 1:3

            if k == 3
                xlabel = string("log<sub>10</sub> m<sub>*</sub>")
            else
                xlabel = nothing
            end

            # 1st column has ylabels
            if i == 1
                ylabel = coeffnames[k]
            else
                ylabel = nothing
            end

            if i != 2

                # create dataframe for violin plot
                plotdf = DataFrame(data = [], group = [])
                for j in 1:4
                    nextdf = DataFrame(data = rand(csamps[j, k], 8000),
                                       group = massrange[j])
                    plotdf = vcat(plotdf, nextdf)
                end

                subplots[k, i] = plot(plotdf,
                                      x = :group,
                                      y = :data,
                                      Geom.violin, #(trim = false),
                                      Theme(default_color=varcolors[var_num]),
                                      #Coord.Cartesian(ymin=-5, ymax=5),
                                      Guide.xlabel(xlabel), Guide.ylabel(ylabel))
                                      #Theme(key_position = :none))

            else

                varinds = [1, 2, 3]
                deleteat!(varinds, var_num)

                plotdf = DataFrame(data = [], group = [])
                types = Array{String,1}()

                for z in 1:2

                    otherind = varinds[z]

                    for j in 1:4

                        nextdf = DataFrame(data = rand(csamps[z, j, k], 8000),
                                           group = massrange[j])
                        plotdf = vcat(plotdf, nextdf)

                        append!(types, fill(varnames[otherind], 8000))
                                            #length(csamps[z, j, k])))
                    end
                end

                #println(size(types))
                #println(size(plotdf))

                subplots[k, i] = plot(plotdf,
                                      x = :group,
                                      y = :data,
                                      color = types,
                                      Geom.violin) #(trim = false))
                                      #Theme(default_color=midcolors[z])
                                      #Scale.color_discrete_manual(midcolors...),
                                      #Coord.Cartesian(ymin=-5, ymax=5),
                                      #Guide.xlabel(xlabel), Guide.ylabel(ylabel))
                                      #Theme(key_position = :none))

            end

        end
    end

    gridstack(subplots)
end
