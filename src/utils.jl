module CenQuenUtils

import Gadfly.plot, Gadfly.Geom
export plot_binned_errs

function get_errbar(binpts)

    npts = length(binpts)
    ymean = mean(binpts)
    ystd = std(binpts)
    yerr = ystd / sqrt(npts)

    return ymean, yerr
end


function plot_binned_errs(x, y, bins)

    xedges = linspace(minimum(x), maximum(x), bins + 1)
    xmeans = [mean(x[xedges[i] .<= x .<= xedges[i + 1]]) for i in 1:bins]

    ymeans = []
    ymins = []
    ymaxs = []

    for i in 1:bins
        ymean, yerr = get_errbar(y[xedges[i] .<= x .<= xedges[i + 1]])
        append!(ymeans, ymean)
        append!(ymins, ymean - yerr)
        append!(ymaxs, ymean + yerr)
    end

    plot(x=xmeans, y=ymeans, ymin=ymins, ymax=ymaxs, Geom.point, Geom.errorbar)
end

end
