using Plots

@views function acoustic_2D()
    # Physics
    Lx, Ly = 10.0, 10.0
    ρ      = 1.0
    K      = 1.0
    μ      = 1.0
    ttot   = 20.0
    # Numerics
    nx, ny = 128, 128
    nout   = 10
    # Derived numerics
    dx, dy = Lx/nx, Ly/ny
    dt     = min(dx,dy)/sqrt((K + 4/3*μ)/ρ)/2.1
    nt     = cld(ttot, dt)
    xc, yc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    # Array initialisation
    P      =  exp.(.-(xc .- Lx/2).^2 .-(yc' .- Ly/2).^2)
    dPdt   = zeros(Float64, nx  ,ny  )
    dVxdt  = zeros(Float64, nx-1,ny-2)
    dVydt  = zeros(Float64, nx-2,ny-1)
    Vx     = zeros(Float64, nx+1,ny  )
    Vy     = zeros(Float64, nx  ,ny+1)
    qVxx   = zeros(Float64, nx  ,ny  )
    qVyy   = zeros(Float64, nx  ,ny  )
    qVxy   = zeros(Float64, nx-1,ny-1)
    ∇V     = zeros(Float64, nx  ,ny  )
    # Time loop
    anim = Animation()
    for it = 1:nt
        dPdt    .= .-K.*(∇V)
        P       .= P .+ dt.*dPdt
        qVxx    .= qVxx .+ dt.*(2.0 .* μ .* (diff(Vx,dims=1)/dx .- 1/3 .* ∇V))
        qVyy    .= qVyy .+ dt.*(2.0.* μ .* (diff(Vy,dims=2)/dy .- 1/3 .* ∇V))
        qVxy    .= qVxy .+ dt.*(diff(Vx[2:end-1,:],dims=2) .+ diff(Vy[:,2:end-1],dims=1))
        dVxdt   .= .-1.0./ρ.*diff(P[:,2:end-1],dims=1)./dx .+ 
            (diff(qVxx[:,2:end-1],dims=1)./dx) .+
            (diff(qVxy,dims=2)./dy)
        dVydt   .= .-1.0./ρ.*diff(P[2:end-1,:],dims=2)./dy .+
            (diff(qVyy[2:end-1,:],dims=2)./dy) .+
            (diff(qVxy,dims=1)./dx) 
        Vx[2:end-1,2:end-1]      .= Vx[2:end-1,2:end-1] .+ dt.*dVxdt
        Vy[2:end-1,2:end-1]      .= Vy[2:end-1,2:end-1] .+ dt.*dVydt
        ∇V      .= diff(Vx,dims=1)./dx .+ diff(Vy,dims=2)./dy
        if it % nout == 0
            opts = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.25, 0.25), c=:davos, xlabel="Lx", ylabel="Ly", title="time = $(round(it*dt, sigdigits=3))")
            frame(anim, heatmap(xc, yc, P'; opts...))
        end
    end
    gif(anim, "elastic_2d.gif", fps=10)
    return
end

acoustic_2D()
