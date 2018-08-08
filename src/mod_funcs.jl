using CSV
using DataFrames
using NearestNeighbors

# get halotools to build galaxy mocks
# first two lines are necessary to get python environment with halotools
ENV["PYTHON"] = "/home/kilian/.envs/astro3/bin/python"
rm(Pkg.dir("PyCall","deps","PYTHON")); Pkg.checkout("PyCall"); Pkg.build("PyCall")
using PyCall

@pyimport astropy.cosmology as astrocosm
@pyimport halotools.empirical_models as halomodels
@pyimport halotools.sim_manager as simman
@pyimport halotools.mock_observables as mockobs


"""
This function loads the halo data from the Chinchilla simulation.
It also estimates the peak halo mass by finding the maximum mass from the
previous snapshots of the simulation. It then adds a column for baryon mass
using the fraction of peak mass as in Moster et al.
"""
function load_halodat()

    # get the data directory of the halo files
    halodir = join(split(Base.@__FILE__, "/")[1:(end - 2)], "/")
    halodir = string(halodir, "/dat/halodat")

    halocols = [:halo_id, :halo_upid, :x, :y, :z, :vx, :vy, :vz, :hfoo1,
                :hfoo2, :halo_mvir, :halo_mz_05, :halo_mz_1, :halo_mz_2,
                :halo_mz_4, :halo_mz_8, :halo_ac, :halo_ac_mpeak,
                :halo_nfw_conc, :halo_spin, :hfoo3, :halo_rvir]
    #            :halo_density_m19]
    halodf = CSV.read(string(halodir, "/halo_merge_rvir.dat"),
                      delim=' ', header=false)
    names!(halodf, halocols)

    halodf[:logMh] = log10.(halodf[:halo_mvir])

    nhalos = size(halodf)[1]

    # need to grab peak halomasses for Moster model of galaxy mass
    halodf[:Mpeak] = [maximum([halodf[:halo_mvir][i],
                               halodf[:halo_mz_05][i],
                               halodf[:halo_mz_1][i],
                               halodf[:halo_mz_2][i],
                               halodf[:halo_mz_4][i],
                               halodf[:halo_mz_8][i]]) for i in 1:nhalos]
    halodf[:logMpeak] = log10.(halodf[:Mpeak])
    halodf[:Mbar] = 0.156 * halodf[:Mpeak]  # baryon mass from Moster et al.

    halodf
end


"""
This function creates a galaxy mock from the halo catalog, given a set of
HOD parameters. It returns the galaxy positions, their host halo id, and
a signifier for whether they are centrals or not.
"""
function get_mock(paramdict, m_th)

    # Set the cosmology to match the Chinchilla simulation for galaxy mocks
    const cosmo = astrocosm.FlatLambdaCDM(H0=100, Om0=0.286)

    # set the halo catalog for making mocks
    cat = simman.CachedHaloCatalog(simname="chinchilla",
                                   redshift=0.0,
                                   version_name="v1",
                                   halo_finder="rockstar")

    # build the HOD model with the given parameters
    model = halomodels.PrebuiltHodModelFactory("zheng07",
                                               redshift=0.0,
                                               modulate_with_cenocc=true,
                                               threshold=m_th)

    model[:param_dict] = paramdict
    model[:populate_mock](cat)

    x = model[:mock][:galaxy_table][:columns]["x"]
    y = model[:mock][:galaxy_table][:columns]["y"]
    z = model[:mock][:galaxy_table][:columns]["z"]
    vz = model[:mock][:galaxy_table][:columns]["vz"]

    pos = mockobs.return_xyz_formatted_array(x, y, z,
                                             period=400.0,
                                             velocity=vz,
                                             velocity_distortion_dimension="z",
                                             redshift=0.0,
                                             cosmology=cosmo)

    # create a column for central galaxies
    cens = [a == "centrals" ? 1 : 0 for a in model[:mock][:galaxy_table][:columns]["gal_type"]]

    # and return an array
    galsarr = zeros(length(cens), 5)
    galsarr[:, 1] = model[:mock][:galaxy_table][:columns]["halo_hostid"]
    galsarr[:, 2:4] = pos
    galsarr[:, 5] = cens

    return galsarr
end


"""
This function gets the galaxy environment densities from a mock.
"""
function get_rho(boxpts::Array{Float64, 2}, Lbox::Float64, spherad::Float64)

    # have to add a 10 Mpc "shell" to account for periodic boundary conditions
    treeshellpts = deepcopy(boxpts)

    # add a periodic shell to the cube sides, one dimension at a time
    # the extra shells of each dimension will add on successive loops
    # to leave a full cube
    for i in 1:3
        farshell = treeshellpts[(treeshellpts[:, i] .< spherad), :]
        farshell[:, i] += Lbox
        nearshell = treeshellpts[(treeshellpts[:, i] .> Lbox - spherad), :]
        nearshell[:, i] -= Lbox
        treeshellpts = vcat(treeshellpts, nearshell, farshell)
    end

    tree = KDTree(transpose(treeshellpts); leafsize=10)

    # set up for the delta calculation
    deltas = zeros(size(boxpts)[1])
    spherevol = (4π / 3) * (spherad^3)
    volfac = (Lbox^3 / size(boxpts)[1]) / spherevol

    # and run the tree search
    for i in 1:(size(boxpts)[1])
        idxs = inrange(tree, boxpts[i, :], 10, false)
        deltas[i] = volfac * length(idxs) - 1
    end

    deltas
end


"""
This function randomly removes galaxies from specified mass bins in a mock
in order to match the quenched fraction specified for those bins.
"""
function get_quenfrac(mock, massbins, fracs)

    nbins = length(fracs)
    keepmock = DataFrame()

    for i in 1:nbins

        bindf = mock[massbins[i] .<= mock[:log10M] .< massbins[i + 1], :]
        ngals = size(bindf)[1]
        nkeep = Int(round(ngals * fracs[i]))
        keepinds = randperm(ngals)[1:nkeep]
        bindf = bindf[keepinds, :]

        if i == 1
            keepmock = bindf
        else
            keepmock = [keepmock; bindf]
        end
    end

    keepmock
end


"""
This function computes the baryon conversion efficiency of a halo using the
model from Moster et al.
"""
function bconv_eff(M::Float64, M₁::Float64, ϵ::Float64, β::Float64, γ::Float64)

    ϵ * ((M / M₁)^(-β) + (M / M₁)^γ)^(-1)
end


"""
This function generates a galaxy stellar mass for each halo in an array.
"""
function gen_mstars(mpeak::Array{Float64, 1}, mbar::Array{Float64, 1},
                    params::Array{Float64, 1}; fix_sig::Bool=true)

    nhalos = length(mpeak)

    if fix_sig  # default is fixed scatter
        M₁, ϵ, β, γ = params
    else
        M₁, ϵ, β, γ, M_σ, σ₀, α_σ = params
    end

    ϵ_means = bconv_eff.(mpeak, 10.0^M₁, ϵ, β, γ)

    if !fix_sig
        # keep this fixed
        #α_σ = 1.0
        ϵ_scats = σ₀ + log10.((mpeak / 10^M_σ).^(-α_σ) + 1.0)
        effdraws = (randn(nhalos) .* ϵ_scats) .+ log10.(ϵ_means)
        mstars = mbar .* (10 .^ effdraws)
    else
        logms = log10.(mbar .* ϵ_means)
        scatms = logms .+ (0.2 * randn(nhalos))
        mstars = 10 .^ scatms
    end

    mstars
end


"""
This function returns a histogram of counts in the variable of an array vararr
for bin edges defined by bins.
"""
function histbincounts(vararr, bins)

    nbins = length(bins) - 1
    bindat = Array{Int}(nbins)

    for i in 1:nbins
        bindat[i] = sum(bins[i] .< vararr .< bins[i + 1])
    end

    bindat
end


"""
This function generates stellar masses using gen_mstars and then finds the
histogram counts for the massbins passed to it. It first generates a stellar
mass for a set of peak halo masses. Then it takes the halo indices for a set
of galaxy mocks corresponding to specific massbins and finds the number of
galaxies in those mass bins, returning the histogram over all mass bins.
"""
function massbincounts(mpeak::Array{Array{Float64, 1}},
                       mbar::Array{Array{Float64, 1}},
                       massbins::Array{Array{Float64, 1}},
                       params::Array{Float64, 1};
                       fix_sig::Bool=true)

    dathist = zeros(sum([length(mbs) - 1 for mbs in massbins]))

    j = 1
    k = 0
    for i in 1:length(massbins)
        k = k + length(massbins[i]) - 1

        mstars = gen_mstars(mpeak[i], mbar[i],
                            params, fix_sig = fix_sig)

        # count the galaxies in the mass bins from each of the mocks
        dathist[j:k] = histbincounts(log10.(mstars), massbins[i])
        j = k + 1
    end

    dathist
end
