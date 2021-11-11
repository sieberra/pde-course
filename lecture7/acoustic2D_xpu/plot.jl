using Plots, BSON


T_PEAK = 256.10775204017165 # Peak Performance of Titan Xm GPU

function load_data() 
    results = BSON.load("scaling_acc2D.bson")

    perf_results = results[":Perf"]
    base_results = results[":Base"]

    return base_results, perf_results
end

function do_plot()
    base_results, perf_results = load_data()

    map_f = (a) -> begin
        i,T = a
        return ((16*2^i)^2, T)
    end

    base_results = map(map_f, base_results)
    perf_results = map(map_f, perf_results)

    plot(base_results, label="acoustic_2d_xpu")
    plot!(perf_results, label="acoustic_2D_perf_xpu", 
        ann=[(1e4,200,text("T_peak = $(round(T_PEAK, sigdigits=3)) GB/s",pointsize=12)),
        (1e4,220,text("Arith Prec. = Float64",pointsize=12))],
         legend=:bottomright)


    xlabel!("# of Gridpoints")
    ylabel!("T_eff in GB/s")
    xaxis!(:log)
    title!("Results for Acoustics 2D")

    savefig("scaling_acc2D.png")

    return (base_results, perf_results)
end