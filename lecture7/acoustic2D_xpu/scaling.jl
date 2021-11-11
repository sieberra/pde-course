using Plots, Printf, CUDA
using BSON

const USE_GPU = true
using ParallelStencil
using ParallelStencil.FiniteDifferences2D

@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

@parallel function compute_V!(Vx,Vy,P,dx,dy,ρ,dt)
    @all(Vx) = @all(Vx) - dt/ρ * @d_xi(P)/dx
    @all(Vy) = @all(Vy) - dt/ρ * @d_yi(P)/dy
    return nothing
end

@parallel function compute_P!(P,Vx,Vy,dt,K,dx,dy)
    @inn(P) = @inn(P) - dt*K*(@d_xa(Vx)/dx + @d_ya(Vy)/dy)
    return nothing
end

@views function acoustic_2D_base(;do_visu=false,scaling=1,the_time=100)
    # Physics
    Lx, Ly = 10.0, 10.0
    ρ = 1.0
    K = 1.0
    ttot = the_time
    # Numerics
    nx, ny = 16*2^scaling, 16*2^scaling
    nout = 50
    # Derived Numerics
    dx, dy = Lx/nx, Ly/ny
    dt = min(dx,dy)/sqrt(K/ρ)/2.1
    nt = cld(ttot,dt)
    xc, yc = LinRange(dx/2, Lx-dx/2,nx), LinRange(dy/2, Ly-dy/2,ny)
    # Array initialisation
    P = Data.Array(exp.(.-(xc .- Lx/2).^2  .-(yc' .- Ly/2).^2 ))
    Vx = @zeros(nx-1, ny-2)
    Vy = @zeros(nx-2, ny-1)


    #Time Loop
    t_tic = 0.0; niter = 0
    for it = 1:nt
        if (it==11) t_tic = Base.time(); niter = 0 end
        @parallel compute_V!(Vx,Vy,P,dx,dy,ρ,dt)
        @parallel compute_P!(P,Vx,Vy,dt,K,dx,dy)
        niter += 1
        if do_visu && (it % nout == 0)
            opts = (aspect_ratio=1, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]),
            clims=(-0.25,0.25), c=:davos, xlabel="Lx", ylabel="Ly", 
            title = "time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc,yc,P'; opts...))
        end
    end

    t_toc = Base.time() - t_tic
    A_eff = (3*2)/1e9*nx*ny*sizeof(Float64)
    t_it = t_toc/niter
    T_eff = A_eff/t_it 
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, 
        round(T_eff,sigdigits=3), niter)
    return T_eff
end



@parallel_indices (ix,iy) function compute_V_idx!(Vx,Vy,P,dt_ρ_dx,dt_ρ_dy,nx,ny)
    if ix <= nx-1 && iy <= ny-2
        Vx[ix,iy] = Vx[ix,iy]  - dt_ρ_dx * (P[ix+1,iy+1] - P[ix,iy+1])
    end

    if ix <= nx-2 && iy <= ny-1
        Vy[ix,iy] = Vy[ix,iy] - dt_ρ_dy * (P[ix+1,iy+1] - P[ix+1,iy])
    end
    return nothing
end

@parallel_indices (ix,iy) function compute_P_idx!(P,Vx,Vy,dtK,_dx,_dy,nx,ny)
    if ix <= nx-2 && iy <= ny-2
        P[ix+1,iy+1] = P[ix+1,iy+1] - dtK*((Vx[ix+1,iy] - Vx[ix,iy])*_dx + 
        (Vy[ix,iy+1] - Vy[ix,iy])*_dy)
    end
    return nothing
end

@views function acoustic_2D_perf(;do_visu=false,scaling=1,the_time=100)
    # Physics
    Lx, Ly = 10.0, 10.0
    ρ = 1.0
    K = 1.0
    ttot = the_time
    # Numerics
    nx, ny = 16*2^scaling, 16*2^scaling
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
        @parallel compute_V_idx!(Vx,Vy,P,dt_ρ_dx,dt_ρ_dy,nx,ny)
        @parallel compute_P_idx!(P,Vx,Vy,dtK,_dx,_dy,nx,ny)
        niter += 1
        if do_visu && (it % nout == 0)
            opts = (aspect_ratio=1, xlims=(xc[1],xc[end]), ylims=(yc[1],yc[end]),
            clims=(-0.25,0.25), c=:davos, xlabel="Lx", ylabel="Ly", 
            title = "time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc,yc,P'; opts...))
        end
    end

    t_toc = Base.time() - t_tic
    A_eff = 9/1e9*nx*ny*sizeof(Float64)
    t_it = t_toc/niter
    T_eff = A_eff/t_it 
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, 
    round(T_eff,sigdigits=3), niter)
    return T_eff
end


function do_scaling_exp() 
    result_base = []
    result_perf = []

    times = [5000,5000,2000, 500, 100, 25, 4, 1]

    for i = 1:8
        Teff_base = acoustic_2D_base(scaling=i,the_time=times[i])
        Teff_perf = acoustic_2D_perf(scaling=i,the_time=times[i])
        push!(result_base, (i,Teff_base))
        push!(result_perf, (i,Teff_perf))
    end

    BSON.bson("scaling_acc2D.bson", Dict(":Base" => result_base, ":Perf" => result_perf))
end



