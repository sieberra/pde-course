using Plots, Printf, CUDA

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end


@parallel_indices (ix,iy) function compute_V!(Vx,Vy,P,dt_ρ_dx,dt_ρ_dy,nx,ny)
    if ix <= nx-1 && iy <= ny-2
        Vx[ix,iy] = Vx[ix,iy]  - dt_ρ_dx * (P[ix+1,iy+1] - P[ix,iy+1])
    end

    if ix <= nx-2 && iy <= ny-1
        Vy[ix,iy] = Vy[ix,iy] - dt_ρ_dy * (P[ix+1,iy+1] - P[ix+1,iy])
    end
    return nothing
end

@parallel_indices (ix,iy) function compute_P!(P,Vx,Vy,dtK,_dx,_dy,nx,ny)
    if ix <= nx-2 && iy <= ny-2
        P[ix+1,iy+1] = P[ix+1,iy+1] - dtK*((Vx[ix+1,iy] - Vx[ix,iy])*_dx + 
        (Vy[ix,iy+1] - Vy[ix,iy])*_dy)
    end
    return nothing
end

@views function acoustic_2D(;do_visu=false)
    # Physics
    Lx, Ly = 10.0, 10.0
    ρ = 1.0
    K = 1.0
    ttot = 1e2
    # Numerics
    nx, ny = 32*4, 32*4
    nout = 50
    # Derived Numerics
    dx, dy = Lx/nx, Ly/ny 
    dt = min(dx,dy)/sqrt(K/ρ)/2.1
    nt = cld(ttot,dt)
    xc, yc = LinRange(dx/2, Lx-dx/2,nx), LinRange(dy/2, Ly-dy/2,ny)
    dt_ρ_dx = dt/ρ/dx
    dt_ρ_dy = dt/ρ/dy
    dtK = dt*K
    _dx , _dy = 1.0/dx, 1.0/dy
    # Array initialisation
    P = Data.Array(exp.(.-(xc .- Lx/2).^2  .-(yc' .- Ly/2).^2 ))
    Vx = @zeros(nx-1, ny-2)
    Vy = @zeros(nx-2, ny-1)


    #Time Loop
    t_tic = 0.0; niter = 0
    for it = 1:nt
        if (it==11) t_tic = Base.time(); niter = 0 end
        @parallel compute_V!(Vx,Vy,P,dt_ρ_dx,dt_ρ_dy,nx,ny)
        @parallel compute_P!(P,Vx,Vy,dtK,_dx,_dy,nx,ny)
        niter += 1
        if do_visu && (it % nout == 0)
            opts = (aspect_ratio=1, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]),
            clims=(-0.25,0.25), c=:davos, xlabel="Lx", ylabel="Ly", 
            title = "time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc,yc,P'; opts...))
        end
    end

    t_toc = Base.time() - t_tic
    A_eff = 10/1e9*nx*ny*sizeof(Float64)
    t_it = t_toc/niter
    T_eff = A_eff/t_it 
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, 
    round(T_eff,sigdigits=3), niter)
    return
end

acoustic_2D(; do_visu=false)