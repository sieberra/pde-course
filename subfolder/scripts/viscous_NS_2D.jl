using Plots

@views function acoustic_2D(;do_vis=false)
    # Physics
    Lx, Ly = 10.0, 10.0
    ρ      = 1.0
    K      = 1.0
    μ      = 1.0
    ttot   = 5.0
    # Numerics
    nx, ny = 128, 128
    nout   = 50
    # Derived numerics
    dx, dy = Lx/nx, Ly/ny
    dt     = min(dx,dy)^2/(K + 4/3*μ)/ρ/4.1
    nt     = cld(ttot, dt)
    xc, yc = LinRange(dx/2, Lx-dx/2, nx), LinRange(dy/2, Ly-dy/2, ny)
    # Array initialisation
    P      =  exp.(.-(xc .- Lx/2).^2 .-(yc' .- Ly/2).^2)
    dPdt   = zeros(Float64, nx  ,ny  )
    dVxdt  = zeros(Float64, nx-1,ny-2)
    dVydt  = zeros(Float64, nx-2,ny-1)
    Vx     = zeros(Float64, nx+1,ny  )
    Vy     = zeros(Float64, nx  ,ny+1)
    ∇V     = zeros(Float64, nx  ,ny  )
    # Time loop
    anim = nothing
    anim2 = nothing
    if do_vis
        ENV["GKSwstype"]="nul"
        anim = Animation()
        anim2 = Animation()
    end
    for it = 1:nt
        dPdt .= (-K).*(∇V)
        P .= P .+ dPdt .* dt
        #τxx (nx,ny)
        τxx = μ .* (2 .* diff(Vx,dims=1)./dx .- (1/3) .* ∇V)
        #τxy (nx-1,ny-1)
        τxy = μ .* (diff(Vy[:,2:end-1],dims=1)./dx .+ diff(Vx[2:end-1,:],dims=2)./dy)
        #τyx (nx-1,ny-1)
        τyx = μ .* (diff(Vx[2:end-1,:], dims=2)./dy .+ diff(Vy[:,2:end-1],dims=1)./dx)
        #τyy (nx,ny)
        τyy = μ .* (2 .* diff(Vy,dims=2)./dy .- (1/3) .* ∇V)
        dVxdt .= (1/ρ) .* (diff((τxx[:,2:end-1] .- P[:,2:end-1]), dims=1)./dx) .+ 
        (1/ρ) .* (diff((τxy), dims=2)./dy)
        dVydt .= (1/ρ) .* (diff((τyx), dims=1)./dx) .+ 
            (1/ρ) .* (diff((τyy[2:end-1,:] .- P[2:end-1,:]), dims=2)./dy) 

        ∇V .= (diff(Vx,dims=1)./dx) .+ (diff(Vy, dims=2)./dy)
        
        Vx[2:end-1,2:end-1] .= Vx[2:end-1,2:end-1] .+ dt .* dVxdt
        Vy[2:end-1,2:end-1] .= Vy[2:end-1,2:end-1] .+ dt .* dVydt

        if do_vis && it % nout == 0
            opts1 = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.25, 0.25), c=:davos, xlabel="Lx", ylabel="Ly", title="P at time = $(round(it*dt, sigdigits=3))")
            opts2 = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.1, 0.1), c=:davos, xlabel="Lx", ylabel="Ly", title="Vx at time = $(round(it*dt, sigdigits=3))")
            opts3 = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.1, 0.1), c=:davos, xlabel="Lx", ylabel="Ly", title=" τxx at time = $(round(it*dt, sigdigits=3))")
            opts4 = (aspect_ratio=1, xlims=(xc[1], xc[end]), ylims=(yc[1], yc[end]), clims=(-0.1, 0.1), c=:davos, xlabel="Lx", ylabel="Ly", title="time τxy = $(round(it*dt, sigdigits=3))")
            frame(anim, heatmap(xc, yc, P'; opts1...))
            frame(anim2, plot(heatmap(xc,yc, P'; opts1...), 
            heatmap(xc,yc,Vx[1:end-1,:]'; opts2...),
            heatmap(xc,yc,τxx'; opts3...),
            heatmap(xc,yc,τxy'; opts4...)))
        end
    end
    if do_vis 
        gif(anim, "viscous_2d.gif", fps=10)
        gif(anim2, "viscous_2d_2.gif", fps=10)
    end
    return
end

acoustic_2D(;do_vis=true)
