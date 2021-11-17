using Plots

function do_plot()
    base_time = 1.9346991 # Base time
    gpu_times = [1.9710360, 4.0661600, 6.1612659, 4.2916641]
    gpu_x = [1,2,3,4]

    final_map = map((i) -> 
    begin
        x,y = i
        y = base_time/y
        return (x,y)
    end, zip(gpu_x, gpu_times))

    plot(final_map, title="Parallel Efficiency", xlabel="Num of GPUs", 
        ylabel="Efficiency", legend=:topright, label="Efficiency")
    savefig("scaling_results_multigpu.png")
end

do_plot()