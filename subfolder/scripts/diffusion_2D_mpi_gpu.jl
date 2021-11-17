# # 2D linear diffusion Julia MPI solver
# run: ~/.julia/bin/mpiexecjl -n 4 julia --project diffusion_2D_mpi.jl
using Plots, Printf, MAT
using CUDA
import MPI

# enable plotting by default
if !@isdefined do_save; do_save = true end

# MPI functions
@views function update_halo(A, neighbors_x, neighbors_y, comm)
    # Send to / receive from neighbor 1 in dimension x ("left neighbor")
    if neighbors_x[1] != MPI.MPI_PROC_NULL
        # ...
        sendbuf = zeros(size(A[2, :]))
        recvbuf = zeros(size(A[1,:]))
        copyto!(sendbuf,A[2,:])
        MPI.Send(sendbuf, neighbors_x[1], 1, comm)
        MPI.Recv!(recvbuf, neighbors_x[1], 2, comm)
        copyto!(A[1,:], recvbuf)
    end
    # Send to / receive from neighbor 2 in dimension x ("right neighbor")
    if neighbors_x[2] != MPI.MPI_PROC_NULL
        # ...
        sendbuf = zeros(size(A[end-1, :]))
        recvbuf = zeros(size(A[end,:]))
        copyto!(sendbuf, A[end-1,:])
        MPI.Send(sendbuf, neighbors_x[2], 2, comm)
        MPI.Recv!(recvbuf, neighbors_x[2],1, comm)
        copyto!(A[end,:], recvbuf)
    end
    # Send to / receive from neighbor 1 in dimension y ("bottom neighbor")
    if neighbors_y[1] != MPI.MPI_PROC_NULL
        # ...
        sendbuf = zeros(size(A[:,2]))
        recvbuf = zeros(size(A[:,1]))
        copyto!(sendbuf, A[:,2])
        MPI.Send(sendbuf, neighbors_y[1], 3, comm)
        MPI.Recv!(recvbuf, neighbors_y[1], 4, comm)
        copyto!(A[:,1], recvbuf)
    end
    # Send to / receive from neighbor 2 in dimension y ("top neighbor")
    if neighbors_y[2] != MPI.MPI_PROC_NULL
        # ...
        sendbuf = zeros(size(A[:,end-1]))
        recvbuf = zeros(size(A[:,end]))
        copyto!(sendbuf, A[:,end-1])
        MPI.Send(sendbuf, neighbors_y[2], 4, comm)
        MPI.Recv!(recvbuf, neighbors_y[2], 3, comm)
        copyto!(A[:,end], recvbuf)
    end
    return
end



function update_step!(C,qx,qy, D, dt, dx, dy)
    qx  .= .-D.*(C[1:end-1, 2:end-1] .- C[2:end,2:end-1])./dx
    qy  .= .-D.*(C[2:end-1,1:end-1] .- C[2:end-1, 2:end])./dy
    C[2:end-1,2:end-1] .= C[2:end-1,2:end-1] .- dt.*((qx[1:end-1,:] .- qx[2:end,:])./dx .+ 
        (qy[:,1:end-1] .- qy[:,2:end])./dy)
end

@views function diffusion_2D_mpi(; do_save=false)
    # MPI
    MPI.Init()
    dims        = [0,0]
    comm        = MPI.COMM_WORLD
    nprocs      = MPI.Comm_size(comm)
    MPI.Dims_create!(nprocs, dims)
    comm_cart   = MPI.Cart_create(comm, dims, [0,0], 1)
    me          = MPI.Comm_rank(comm_cart)
    coords      = MPI.Cart_coords(comm_cart)
    neighbors_x = MPI.Cart_shift(comm_cart, 0, 1)
    neighbors_y = MPI.Cart_shift(comm_cart, 1, 1)

    # GPU-specific for MPI
    comm_l = MPI.Comm_split_type(comm, MPI.MPI_COMM_TYPE_SHARED, me)
    me_l = MPI.Comm_rank(comm_l)
    GPU_ID = CUDA.device!(me_l)

    if (me==0) println("nprocs=$(nprocs), dims[1]=$(dims[1]), dims[2]=$(dims[2])") end
    # Physics
    lx, ly     = 10.0, 10.0
    D          = 1.0
    nt         = 100
    # Numerics
    nx, ny     = 32, 32                             # local number of grid points
    nx_g, ny_g = dims[1]*(nx-2)+2, dims[2]*(ny-2)+2 # global number of grid points
    # Derived numerics
    dx, dy     = lx/nx_g, ly/ny_g                   # global
    dt         = min(dx,dy)^2/D/4.1
    # Array allocation
    qx         = CUDA.zeros(Float64, nx-1,ny-2)
    qy         = CUDA.zeros(Float64, nx-2,ny-1)
    # Initial condition
    x0, y0     = coords[1]*(nx-2)*dx, coords[2]*(ny-2)*dy
    xc         = [x0 + ix*dx - dx/2 - 0.5*lx  for ix=1:nx]
    yc         = [y0 + iy*dy - dy/2 - 0.5*ly  for iy=1:ny]
    C_init     = collect(exp.(.-xc.^2 .-yc'.^2))
    C          = CuArray(copy(C_init))

    # GPU-specific stuff
    THREADSX = 32
    THREADSY = 32
    BLOCKSX = 1 
    BLOCKSY = 1
    cuthreads = (THREADSX,THREADSY)
    cublocks = (BLOCKSX,BLOCKSY)

    t_tic = 0.0
    # Time loop
    for it = 1:nt
        if (it==11) t_tic = Base.time() end
        size_1 = size(C,1)
        size_2 = size(C,2)
        
        update_step!(C,qx,qy,D,dt,dx,dy)
        synchronize()
        update_halo(C, neighbors_x, neighbors_y, comm_cart)

        if it % 1 == 0
            if do_save file = matopen("$(@__DIR__)/mpi2D_gpu_out_C_$(me)_it_$(it).mat", "w"); write(file, "C", Array(C)); close(file) end
        end
    end
    t_toc = (Base.time()-t_tic)
    if (me==0) @printf("Time = %1.4e s, T_eff = %1.2f GB/s \n", t_toc, round((2/1e9*nx*ny*sizeof(lx))/(t_toc/(nt-10)), sigdigits=2)) end
    # Save to visualise
    if do_save file = matopen("$(@__DIR__)/mpi2D_gpu_out_C_$(me).mat", "w"); write(file, "C", Array(C)); close(file) end
    MPI.Finalize()
    return
end

diffusion_2D_mpi(; do_save=do_save)

