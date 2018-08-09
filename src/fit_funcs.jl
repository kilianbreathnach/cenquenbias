using DataFrames


"""
This function takes an array of arrays of dataframes. The dataframes are
subsets of galaxies within given mass and environment bins. The function then
counts the number of galaxies within each dataframe that fit into bins of
other galaxy properties.
"""
function bincounts(framearr::Array{Array{DataFrames.DataFrame, 1}, 1},
                   rbins, sbins, vbins)

    nmbins = length(framearr)
    ndbins = length(framearr[1])
    nrbins = length(rbins) - 1
    nsbins = length(sbins) - 1
    nvbins = length(vbins) - 1

    datbins = Array{Int}(nmbins * ndbins * nrbins * nsbins * nvbins)

    for (i, mframes) in enumerate(framearr)
        ipos = ndbins * nrbins * nsbins * nvbins * (i - 1)
        for (j, dframe) in enumerate(mframes)
            jpos = ipos + nrbins * nsbins * nvbins * (j - 1)

            for k in 1:nrbins
                kdf = dframe[rbins[k] .<= dframe[:logRe] .< rbins[k + 1], :]
                kpos = jpos + nsbins * nvbins * (k - 1)

                for l in 1:nsbins
                    ldf = kdf[sbins[l] .<= kdf[:logsurf] .< sbins[l + 1], :]
                    lpos = kpos + nvbins * (l - 1)

                    for m in 1:nvbins
                        datbins[lpos + m] = sum(vbins[m] .<= ldf[:logv] .< vbins[m + 1])
                    end
                end
            end
        end
    end
    datbins
end


"""
This function generates values for R_e in a mock dataframe given a set of
model parameters.
"""
function re_func(params::Array{Float64, 1},
                 mockdfs::Array{Array{DataFrames.DataFrame, 1}, 1})

    βr₀, βrm, βrm2, βrmd, βrc, βrs = params
    # σᵣ = 0.02

    for arr in mockdfs
        for df in arr
            # rs = randn(length(df[:logRe]))
            # rs = rs * σᵣ
            rm = βr₀ + ((βrm * df[:logMh]) .+ (βrm2 * (df[:logMh] .^ 2)) .+
                        (βrmd * df[:logMd]) .+
                        (βrc * df[:logc]) .+ (βrs * df[:logspin]))
            df[:logRe] = rm  # .+ rs
        end
    end

#     βr₀, βrm, βrmd, βrc, βrs, βrmmd, βrmc, βrms, σᵣ = params
#
#     for arr in mockdfs
#         for df in arr
#             rs = randn(length(df[:logRe]))
#             rs = rs * σᵣ
#             rm = βr₀ + ((βrm * df[:logMh]) .+ (βrmd * df[:logMd]) .+
#                         (βrc * df[:logc]) .+ (βrs * df[:logspin]) .+
#                         (βrmmd * (df[:logMh] .* df[:logMd])) .+
#                         (βrmc * (df[:logMh] .* df[:logc])) .+
#                         (βrms * (df[:logMh] .* df[:logspin])))
#             df[:logRe] = rm .+ rs
#         end
#     end
end


"""
This function does the same for surface density of a galaxy given a halo.
"""
function surfunc(params::Array{Float64, 1},
                 mockdfs::Array{Array{DataFrames.DataFrame, 1}, 1})

    βs₀, βsm, βsm2, βsmd, βsc, βss = params
    # σₛ = 0.02

    for arr in mockdfs
        for df in arr
            # ss = randn(length(df[:logsurf]))
            # ss = ss * σₛ
            sm = βs₀ + ((βsm * df[:logMh]) .+ (βsm2 * (df[:logMh] .^ 2)) .+
                        (βsmd * df[:logMd]) .+
                        (βsc * df[:logc]) .+ (βss * df[:logspin]))
            df[:logsurf] = sm   # .+ ss
        end
    end

#     βs₀, βsm, βsmd, βsc, βss, βsmmd, βsmc, βsms, σₛ = params[10:end]
#
#     for arr in mockdfs
#         for df in arr
#             ss = randn(length(df[:logsurf]))
#             ss = ss * σₛ
#             sm = βs₀ + ((βsm * df[:logMh]) .+ (βsmd * df[:logMd]) .+
#                         (βsc * df[:logc]) .+ (βss * df[:logspin]) .+
#                         (βsmmd * (df[:logMh] .* df[:logMd])) .+
#                         (βsmc * (df[:logMh] .* df[:logc])) .+
#                         (βsms * (df[:logMh] .* df[:logspin])))
#             df[:logsurf] = sm .+ ss
#         end
#     end
end


"""
And one for velocity dispersion
"""
function v_func(params::Array{Float64, 1},
                 mockdfs::Array{Array{DataFrames.DataFrame, 1}, 1})

    βv₀, βvm, βvm2, βvmd, βvc, βvs = params
    #σᵥ = 0.02

    for arr in mockdfs
        for df in arr
            # vs = randn(length(df[:logv]))
            # vs = vs * σᵥ
            vm = βv₀ + ((βvm * df[:logMh]) .+ (βvm2 * (df[:logMh] .^ 2)) .+
                        (βvmd * df[:logMd]) .+
                        (βvc * df[:logc]) .+ (βvs * df[:logspin]))
            df[:logv] = vm   # .+ vs
        end
    end

#     βv₀, βvm, βvmd, βvc, βvs, σᵥ = params[13:end]
#
#     for arr in mockdfs
#         for df in arr
#             vs = randn(length(df[:logv]))
#             vs = vs * σᵥ
#             vm = βv₀ + ((βvm * df[:logMh]) .+ (βvmd * df[:logMd]) .+ (βvc * df[:logc]) .+ (βvs * df[:logspin]))
#             df[:logv] = vm .+ vs
#         end
#     end
end


"""
This function accepts the model parameters and computes a set of galaxy mock
observations then bins the observed data points into our histogram.
"""
function gen_mod_bincounts(params::Array{Float64, 1},
                           mockdfs::Array{Array{DataFrames.DataFrame, 1}, 1},
                           varswitch::Array{Int, 1},
                           rbins, sbins, vbins)

    # handle combinatorics of the three variable switches
    if varswitch[1] == 1
        re_func(params[1:6], mockdfs)

        if varswitch[2] == 1
            surfunc(params[7:12], mockdfs)

            if varswitch[3] == 1
                v_func(params[13:end], mockdfs)
            end

        elseif varswitch[3] == 1
            v_func(params[7:12], mockdfs)
        end

    elseif varswitch[2] == 1
        surfunc(params[1:6], mockdfs)

        if varswitch[3] == 1
            v_func(params[7:12], mockdfs)
        end

    elseif varswitch[3] == 1
        v_func(params, mockdfs)
    end

    hist = bincounts(mockdfs, rbins, sbins, vbins)
    hist
end


"""
This function evaluates Cash's cstat for two arrays of model and observations.
Since the sum of observed values and the sum of observed values times the log
of observed values can be known in advance, these are passed as an additional
argument. The mask of nonzero values in the data is also passed, in order
to avoid unnecessary summing and unstable values.
"""
function cstat(mod, obs, obsums, nonz)
    modsum = sum(mod)
    return 2 * (modsum - obsums[1] + obsums[2] - sum(obs[nonz] .* log.(mod[nonz])))
end
