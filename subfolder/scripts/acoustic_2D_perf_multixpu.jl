# juliap -O3 --check-bounds=no --math-mode=fast acoustic_2D_perf_gpu.jl
const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end
using Plots, Printf
using ImplicitGlobalGrid


@parallel_indices (ix,iy) function compute_V!(Vx, Vy, P, dt_ρ_dx, dt_ρ_dy)
    if (ix<=size(Vx,1) && iy<=size(Vx,2))
        Vx[ix,iy] = Vx[ix,iy] - dt_ρ_dx*(P[ix+1,iy+1] - P[ix,iy+1])
    end
    if (ix<=size(Vy,1) && iy<=size(Vy,2))
        Vy[ix,iy] = Vy[ix,iy] - dt_ρ_dy*(P[ix+1,iy+1] - P[ix+1,iy])
    end
    return
end

@parallel_indices (ix,iy) function compute_P!(P, Vx, Vy, dtK, _dx, _dy, size_P1_2, size_P2_2)
    if (ix<=size_P1_2 && iy<=size_P2_2)
        P[ix+1,iy+1] = P[ix+1,iy+1] - dtK*((Vx[ix+1,iy] - Vx[ix,iy])*_dx + (Vy[ix,iy+1] - Vy[ix,iy])*_dy)
    end
    return
end

@views function acoustic_2D(; do_visu=false)
    # Physics
    Lx, Ly  = 10.0, 10.0
    ρ       = 1.0
    K       = 1.0
    ttot    = 1e1
    # Numerics
    nx, ny  = 32, 32 # number of grid points
    nout    = 10
    # Derived numerics
    me, dims = init_global_grid(nx,ny, 1)
    @static if USE_GPU select_device() end
    dx, dy  = Lx/nx_g(), Ly/ny_g()
    dt      = min(dx, dy)/sqrt(K/ρ)/2.1
    nt      = cld(ttot, dt)
    xc, yc  = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    dt_ρ_dx = dt/ρ/dx
    dt_ρ_dy = dt/ρ/dy
    dtK     = dt*K
    _dx, _dy= 1.0/dx, 1.0/dy
    # Array initialisation
    P       = @zeros(nx,ny)
    P       .= Data.Array([exp(-(x_g(ix,dx,P)+dx/2 -Lx/2)^2 -(y_g(iy,dy,P)+dy/2 -Ly/2)^2) for ix=1:size(P,1), iy=1:size(P,2)])
    Vx      = @zeros(nx-1,ny-2)
    Vy      = @zeros(nx-2,ny-1)
    size_P1_2, size_P2_2 = size(P,1)-2, size(P,2)-2
    t_tic = 0.0; niter = 0
    # Visualisation stuff
    if do_visu 
        if (me==0) ENV["GKSwstype"]="nul"; if isdir("viz_acc2d_mxpu_out")==false mkdir("viz_acc2d_mxpu_out") end; loadpath = "./viz_acc2d_mxpu_out/"; anim = Animation(loadpath, String[]); println("Animation directory: $(anim.dir)") end
        nx_v, ny_v = (nx-2)*dims[1], (ny-2)*dims[2]
        if (nx_v*ny_v*sizeof(Data.Number) > 0.8*Sys.free_memory()) error("Not enough memory for visualization.") end
        P_v = zeros(nx_v, ny_v)
        P_inn = zeros(nx-2, ny-2)
        Xi_g, Yi_g = LinRange(dx+dx/2, Lx-dx-dx/2, nx_v), LinRange(dy+dy/2, Ly-dy-dy/2, ny_v)
    end
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time(); niter = 0 end
        @parallel compute_V!(Vx, Vy, P, dt_ρ_dx, dt_ρ_dy)
        @hide_communication (8,2) begin
            @parallel compute_P!(P, Vx, Vy, dtK, _dx, _dy, size_P1_2, size_P2_2)
            update_halo!(P)
        end
        niter += 1
        if do_visu && (it % nout == 0)
            P_inn .= P[2:end-1, 2:end-1]; gather!(P_inn, P_v)
            if (me==0)
                opts = (aspect_ratio=1, xlims=(Xi_g[1], Xi_g[end]), ylims=(Yi_g[1], Yi_g[end]), clims=(-0.25, 0.25), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
                heatmap(Xi_g, Yi_g, Array(P_v)'; opts...); frame(anim)
            end
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = (3*2)/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                      # Execution time per iteration [s]
    T_eff = A_eff/t_it                       # Effective memory throughput [GB/s]
    if me==0 @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter) end
    if (do_visu && me == 0) gif(anim,"acoustic_2D_multixpu.gif", fps=5) end
    finalize_global_grid()
    return
end

acoustic_2D(; do_visu=true)
