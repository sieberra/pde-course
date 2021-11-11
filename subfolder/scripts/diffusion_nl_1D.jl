using Test

@views av(A) = 0.5.*(A[1:end-1].+A[2:end])

@views function diffusion_1D()
    # Physics
    Lx   = 10.0
    D0   = 5.0
    n    = 5
    ttot = 2.0
    # Numerics
    nx   = 128
    nout = 50
    # Derived numerics
    dx   = Lx/nx
    xc   = LinRange(dx/2, Lx-dx/2, nx)
    # Array initialisation
    H    =  exp.(.-(xc .- Lx/2).^2)
    Hi   =  copy(H)
    D    = zeros(Float64, nx  )
    dHdt = zeros(Float64, nx-2)
    qx   = zeros(Float64, nx-1)
    # Time loop
    it = 0; t = 0.0
    while t<ttot
        D          .= (D0.*H).^n
        dt          = dx^2/maximum(D)/2.1
        qx         .= .-av(D) .* diff(H)./dx
        dHdt       .= .-diff(qx)./dx
        H[2:end-1] .= H[2:end-1] .+ dt.*dHdt
        it += 1; t += dt
    
    end
    return H, qx
end

H,qx = diffusion_1D();
