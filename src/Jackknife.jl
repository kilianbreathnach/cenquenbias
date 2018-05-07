module Jackknife
using NearestNeighbors

# # to call pymangle (no longer needed since I downloaded the randoms)
# ENV["PYTHON"] = "/home/kilian/.envs/astro2.7/bin/python"
# rm(Pkg.dir("PyCall","deps","PYTHON")); Pkg.checkout("PyCall"); Pkg.build("PyCall")
# using PyCall
# @pyimport pymangle

Pkg.checkout("Distances"); Pkg.build("Distances")
using Distances

export jackmeansvars


"""
This function finds the jackknife subvolume of a sample of galaxies, passed
as an Nx2 array of RA and Dec. The square root of the number of regions can
be passed as nside. The function returns an array of N subvolume indices.
"""
function get_sdss_subvols(radec; nside = 5)

    # get the data directory
    datdir = join(split(abspath(Base.@__FILE__), "/")[1:(end - 2)], "/")
    datdir = string(datdir, "/dat")

    # grab the randoms and put them in an array for the ball tree
    rands = readdlm(string(datdir, "/nyuvagc/lss_random-0.dr72.dat"))
    nrands = size(rands)[1]

    ras = rands[:, 1]
    decs = rands[:, 2]

    randarr = zeros(nrands, 2)
    randarr[:, 1] = ras
    randarr[:, 2] = decs

    # find the 2-d "quantiles" of the randoms,
    # according to the number of sides set for subvolumes
    quantinds = Array{Array{Int}}(nside, nside)
    arrquants = Array{Int}(zeros(nrands))

    raqs = quantile(ras, linspace(0, 1, nside + 1))

    for i in 1:nside

        if i < nside
            rabool = raqs[i] .<= ras .< raqs[i + 1]
        else
            rabool = ras .>= raqs[i]
        end

        dqs = quantile(decs[find(rabool)], linspace(0, 1, nside + 1))

        for j in 1:nside

            if j < nside
                quantinds[i, j] = find(rabool .& (dqs[j] .<= decs .< dqs[j + 1]))
            else
                quantinds[i, j] = find(rabool .& (decs .>= dqs[j]))
            end
            arrquants[quantinds[i, j]] = nside * (i - 1) + j
        end
    end

    # now set up a tree to efficiently search for nearest randoms to the galaxies
    balltree = BallTree(randarr', Distances.Haversine(1.0))
    nninds, dists = knn(balltree, radec', 1)

    # get the subvolume indices of the data
    datquants = arrquants[[n[1] for n in nninds]]

    return datquants
end


"""
This function uses jackknife resampling to compute the mean and variance
of an observable in a dataset. The function to compute the observable is
passed, along with the data to compute it with. The data should be in the
form of an Nxk array or dataframe. This must be the first argument to the
function. Other arguments can be passed as an argument tuple:

obsfunc(data, args...)

another array with the same length as the data is passed containing the indices
of the subvolumes. If the number of subvolumes is less than 5x5, this can be
set with nside. By default, the covariance of the observables are also computed, though this can be changed to calculate only diagonal variances by
setting covar to false. This function is also made to be robust to having
subvolumes that lack observations.
"""
function jackmeansvars(obsfunc::Function, obslen::Int,
                       data, subvolinds::Array{Int, 1}, args=();
                       nside::Int = 5, covar::Bool = true)

    # make an array to hold the subvolume observations
    if obslen > 1
        subobs = Array{Float64}(nside^2, obslen)
    else
        subobs = Array{Float64}(nside^2)
    end
    nvols = nside^2

    # first get the observations
    for i in 1:nvols
        subarr = data[find(subvolinds .== i), :]
        if obslen > 1
            subobs[i, :] = obsfunc(subarr, args...)
        else
            subobs[i] = obsfunc(subarr, args...)
        end
    end

    # in the event that some subvolumes contain none of the datapoints
    finsubvols = .!isnan.(subobs)
    nobsvols = Array{Int}(obslen)
    fininds = Array{Array{Int}}(obslen)

    # this array will hold the jacknife sampled observations
    jackarray = Array{Array{Float64}}(obslen)
    # and this holds the jackknife mean estimates
    jackmeans = Array{Float64}(obslen)

    # get the jackknife observations
    for j in 1:obslen

        # get the number of volumes with finite observation
        # and get their indices
        if obslen > 1
            nobsvols[j] = nobs = sum(finsubvols[:, j])
            fininds[j] = fininds = find(finsubvols[:, j])
            jackarray[j] = zeros(nobsvols)
        else
            nobsvols[j] = nobs = sum(finsubvols)
            fininds[j] = fininds = find(finsubvols)
            jackarray[j] = zeros(nobs)
        end

        volfac = 1 / (nobs - 1)

        for i in 1:nobs
            if obslen > 1
                jackarray[j][i] = volfac * sum(subobs[fininds[find(1:nobs .!= i)], j])
            else
                jackarray[j][i] = volfac * sum(subobs[fininds[find(1:nobs .!= i)]])
            end
        end

        jackmeans[j] = sum(jackarray[j][i]) / nobs
    end

    # and now to construct a covariance matrix
    if obslen > 1
        if covar
            jackcovar = Array{Float64}(obslen, obslen)
        else
            jackcovar = Array{Float64}(obslen)
        end

        for (i, jm) in enumerate(jackmeans)

            if !covar

                idevs = [a - jm for a in subobs[fininds[i], i]]
                ifac = 1 - (1 / nobsvols[i])
                jackcovar[i] = ifac * sum(idevs .^ 2)

            else

                for j in 1:i
                    covarinds = finsubvols[:, i] .& finsubvols[:, j]
                    nfinite = sum(covarinds)
                    volfac = 1 - (1 / nfinite)
                    covarinds = find(covarinds)
                    idevs = [a - jm for a in subobs[covarinds, i]]
                    jdevs = [b - jackmeans[j] for b in subobs[covarinds, j]]
                    jackcovar[i, j] = volfac * sum(idevs .* jdevs)
                    jackcovar[j, i] = jackcovar[i, j]
                end
            end
        end

    # if the observable was a scalar
    else
        jackcovar = volfac * sum((subobs - jackmeans) .^ 2)
        return jackmeans[1], jackcovar
    end

    return jackmeans, jackcovar
end

end
