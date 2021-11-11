# juliap -O3 --check-bounds=no --math-mode=fast diffusion_2D_perf_gpu.jl
using Plots, Printf, CUDA

const USE_GPU = false
using ParallelStencil
using ParallelStencil.FiniteDifferences2D
@static if USE_GPU
    @init_parallel_stencil(CUDA, Float64, 2)
else
    @init_parallel_stencil(Threads, Float64, 2)
end

# macros to avoid array allocation
macro qx(ix,iy)  esc(:( -D_dx*(C[$ix+1,$iy+1] - C[$ix,$iy+1]) )) end
macro qy(ix,iy)  esc(:( -D_dy*(C[$ix+1,$iy+1] - C[$ix+1,$iy]) )) end

function compute!(C2, C, D_dx, D_dy, dt, _dx, _dy, size_C1_2, size_C2_2)
    ix = (blockIdx().x-1) * blockDim().x + threadIdx().x
    iy = (blockIdx().y-1) * blockDim().y + threadIdx().y
    if (ix<=size_C1_2 && iy<=size_C2_2)
        C2[ix+1,iy+1] = C[ix+1,iy+1] - dt*( (@qx(ix+1,iy) - @qx(ix,iy))*_dx + (@qy(ix,iy+1) - @qy(ix,iy))*_dy )
    end
    return
end

@parallel function compute_q!(qx,qy,C,D,dx,dy) 
    @all(qx) = -D*@d_xi(C)/dx
    @all(qy) = -D*@d_yi(C)/dy
    return
end

@parallel function compute_C!(C, qx, qy, dt, dx, dy)
     @inn(C) = @inn(C) - dt * (@d_xa(qx)/dx + @d_ya(qy)/dy)
     return nothing
end

@views function diffusion_2D(; do_visu=false)
    # Physics
    Lx, Ly  = 10.0, 10.0
    D       = 1.0
    ttot    = 1e2
    # Numerics
    nx      = 32*4
    ny      = 32*4
    nout    = 50
    # Derived numerics
    dx, dy  = Lx/nx, Ly/ny
    dt      = min(dx,dy)^2/D/4.1
    nt      = cld(ttot,dt)
    xc, yc  = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    # Array initialisation
    C       = Data.Array(exp.(.-(xc .- Lx/2).^2  .-(yc' .- Ly/2).^2 ))
    qx      = @zeros(nx-1,ny-2)
    qy      = @zeros(nx-2,ny-1)

    t_tic = 0.0; niter = 0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time(); niter = 0 end
        
        @parallel compute_q!(qx, qy, C, D, dx, dy)
        @parallel compute_C!(C, qx, qy, dt, dx, dy)

        niter += 1
        if do_visu && (it % nout == 0)
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(0.0, 1.0), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            display(heatmap(xc, yc, Array(C)'; opts...))
        end
    end
    t_toc = Base.time() - t_tic
    A_eff = 2/1e9*nx*ny*sizeof(Float64)  # Effective main memory access per iteration [GB]
    t_it  = t_toc/niter                  # Execution time per iteration [s]
    T_eff = A_eff/t_it                   # Effective memory throughput [GB/s]
    @printf("Time = %1.3f sec, T_eff = %1.2f GB/s (niter = %d)\n", t_toc, round(T_eff, sigdigits=3), niter)
    return
end

diffusion_2D(; do_visu=false)
