#
# A simple toy atmospheric model that uses upwinded finite differences
# to simulate transport of a puff of non-reactive tracer gas.
#
# I wrote this to teach myself Juila.  
# It's based on what I can remember of my PhD disseration from 12 years ago.
# This is absolutely _not_ how you go about solving this problem in the real world.
# I've no doubt there are much better packages out there for CFD.  Use those.
#
# Visualization is based on code from
# https://lazarusa.github.io/BeautifulMakie/surfWireLines/volumeTransparent/
#
# John Linford <john@redhpc.com>
#
# BSD 3-Clause License
#

using Dates
using GLMakie

"""
    advec_diff(...)

Advection and diffusion for one grid cell via upwind-biased finite differences.
Concentrations and wind values for the two neighboring cells are required.
"""
function advec_diff(
            Δx::Float64,
            c2l::Float64, 
            c1l::Float64, w1l::Float64,
            c00::Float64, w00::Float64, d00::Float64,
            c1r::Float64, w1r::Float64,
            c2r::Float64)
    wind = (w1l + w00) / 2.0
    if wind > 0
        advecL = 1/6 * ( -c2l + 5.0*c1l + 2.0*c00 )
    else 
        advecL = 1/6 * ( 2.0*c1l + 5.0*c00 - c1r )
    end
    advecL *= wind
    
    wind = (w1r + w00) / 2.0
    if wind > 0.0 
        advecR = 1/6 * ( -c1l + 5.0*c00 + 2.0*c1r )
    else
        advecR = 1/6 * ( 2.0*c00 + 5.0*c1r - c2r )
    end
    advecR *= wind
    
    advec = (advecL - advecR) / Δx

    diffL = d00 * (c1l - c00)
    diffR = d00 * (c00 - c1r)
    diff = (diffL  -  diffR) / Δx^2

    return advec + diff
end


"""
    discretize1d(...)

Calculates advection/diffusion in 1D via `advec_diff` with boundaries wrapped around.

# Arguments
 - `Δx::Float64`: Space delta
 - `Δt::Float64`: Time delta
 - `d::Float64`: Diffusion coefficient
 - `c::Vector{Float64}`: Concentrations in ea
 - `w::Vector{Float64})`: Wind coefficients for each concentration
"""
function discretize1d(
            Δx::Float64,
            Δt::Float64,
            d::Float64,
            c::Vector{Float64},
            w::Vector{Float64})

    n = length(c)
    dcdx = Vector{Float64}(undef, n)

    # Boundaries
    dcdx[1] = advec_diff(Δx,
                         c[n-1],        # 2-left neighbor, wrapped around
                         c[n], w[n],    # 1-left neighbor, wrapped around
                         c[1], w[1], d, # Origin cell
                         c[2], w[2],    # 1-right neighbor
                         c[3])          # 2-right neighbor         
    dcdx[2] = advec_diff(Δx,
                         c[n],          # 2-left neighbor, wrapped around
                         c[1], w[1],    # 1-left neighbor
                         c[2], w[2], d, # Origin cell
                         c[3], w[3],    # 1-right neighbor
                         c[4])          # 2-right neighbor

    # Interior
    for i in 3:n-2
        dcdx[i] = advec_diff(Δx,
                             c[i-2],            # 2-left neighbor
                             c[i-1], w[i-1],    # 1-left neighbor
                             c[i], w[i], d,     # Origin cell
                             c[i+1], w[i+1],    # 1-right neighbor
                             c[i+2])            # 2-right neighbor
    end

    # Boundaries
    dcdx[n-1] = advec_diff(Δx,
                           c[n-3],              # 2-left neighbor
                           c[n-2], w[n-2],      # 1-left neighbor
                           c[n-1], w[n-1], d,   # Origin cell
                           c[n], w[n],          # 1-right neighbor
                           c[1])                # 2-right neighbor, wrapped around
    dcdx[n] = advec_diff(Δx,
                         c[n-2],            # 2-left neighbor
                         c[n-1], w[n-1],    # 1-left neighbor
                         c[n], w[n], d,     # Origin cell
                         c[1], w[1],        # 1-right neighbor, wrapped around
                         c[2])              # 2-right neighbor, wrapped around

    c1 = c + Δt*dcdx
    # Mass loss here, if you care about that sort of thing
    return (c1 .* (c1 .> 0))
end


"""
    transport(...)

Discretization by first order upwind-biased finite differences on a possibly non-uniform grid.
3D discretization is achived via a 1D dimension split: x -> y -> z -> y -> x

# Arguments
 - `Δx`, `Δy`, `Δz`: Grid cell size in `x`, `y`, and `z` dimensions
 - `Δt`: timestep size
 - `dH`: Diffusion coefficient in x and y dimensions
 - `dV`: Diffusion coefficient in z dimension
 - `conc`: Concentration field
 - `wind`: Wind field
"""
function transport(
            Δx::Float64,
            Δy::Float64,
            Δz::Float64, 
            Δt::Float64, 
            dH::Float64,
            dV::Float64,
            conc::Array{Float64,4}, 
            wind::Array{Float64,4})

    nx, ny, nz, nspec = size(conc)
    
    for iy ∈ 1:ny, iz ∈ 1:nz, is ∈ 1:nspec
        conc[:,iy,iz,is] = discretize1d(Δx, Δt/2, dH, conc[:,iy,iz,is], wind[:,iy,iz,1])
    end
    for ix ∈ 1:nx, iz ∈ 1:nz, is ∈ 1:nspec
        conc[ix,:,iz,is] = discretize1d(Δy, Δt/2, dH, conc[ix,:,iz,is], wind[ix,:,iz,2])
    end
    for ix ∈ 1:nx, iy ∈ 1:ny, is ∈ 1:nspec
        conc[ix,iy,:,is] = discretize1d(Δz, Δt, dV, conc[ix,iy,:,is], wind[ix,iy,:,3])
    end
    for ix ∈ 1:nx, iz ∈ 1:nz, is ∈ 1:nspec
        conc[ix,:,iz,is] = discretize1d(Δy, Δt/2, dH, conc[ix,:,iz,is], wind[ix,:,iz,2])
    end
    for iy ∈ 1:ny, iz ∈ 1:nz, is ∈ 1:nspec
        conc[:,iy,iz,is] = discretize1d(Δx, Δt/2, dH, conc[:,iy,iz,is], wind[:,iy,iz,1])
    end
end


"""
    update_plot!(...)

Update Observables to advance the visualization.
"""
function update_plot!(points, colors, conc, spec, maxconc=8.61e9, minconc=1e-6)
    ptmp = Vector{Point3f0}()
    ctmp = Vector{Float64}()
    nx, ny, nz, _ = size(conc)
    for ix in 1:nx, iy in 1:ny, iz in 1:nz
        if conc[ix,iy,iz,spec] > minconc
            push!(ptmp, Point3f0(ix,iy,iz))
            push!(ctmp, conc[ix,iy,iz,spec]/maxconc)
        end
    end
    # Holy perf pain batman!  Don't update those observables until you absolutely have to!
    # One update outside the loop is at least 100x faster than updating in place.
    points[] = ptmp
    colors[] = ctmp
end


# Probably should be a main() function or something....
let
    # Time grid
    tstart = Dates.DateTime(2020, 01, 01)
    tend = tstart + Dates.Hour(24)
    Δt = Dates.Minute(15)

    # Space grid
    nx, ny, nz = 50, 50, 25

    # Grid cell dimensions (m)
    Δx, Δy, Δz = 1000.0, 1000.0, 1000.0

    # Horizontal and vertical diffusion coefficients (m²/s)
    dH, dV = 100.0, 50.0

    # Number of chemical species
    nspec = 1

    # Concentration field
    # Concentrations of the same species are contiguous
    conc = Array{Float64,4}(undef, nx, ny, nz, nspec)
    let
        conc .= 0.0
        # Dummy up a tracer point source in the center of the concentration field
        source_coords = size(conc)[1:3] .÷ 2
        conc[source_coords...] = 8.61e9
    end

    # Wind field
    wind = Array{Float64,4}(undef, nx, ny, nz, 3)
    wind_init = "simple"
    if wind_init == "simple"
        # 0.5m/sec cross wind in x-dimension
        wind[:,:,:,1] .= 0.5
        wind[:,:,:,2] .= 0.0
        wind[:,:,:,3] .= 0.0
    elseif wind_init == "cyclone"
        # Cyclonic wind
        for ix ∈ 1:nx, iy ∈ 1:ny, iz ∈ 1:nz
            wind[ix,iy,iz,:] = 0.5 .* [cos(nx/2 - ix), sin(ny/2 - iy), 0.25]
        end
    else
        error("Invalid wind_init: $wind_init")
    end

    # Temperature field
    # 297.15 °K == 24 °C == 75.2 °F
    temp = Array{Float64,3}(undef, nx, ny, nz)
    temp .= 297.15

    # Visualization
    fig = Figure(resolution=(1200,1200))
    ax = Axis3(fig[1,1], perspectiveness=0.5, azimuth=π/3, elevation=π/6, aspect=:data)
    points = Node(Vector{Point3f0}())
    colors = Node(Vector{Float64}())

    # Initial plot
    limits!(ax, 1, nx, 1, ny, 1, nz)
    update_plot!(points, colors, conc, 1)
    meshscatter!(ax, points, color=colors, colormap=RGBAf0.(to_colormap(:turbo, 10), LinRange(0,1,10)))

    # Time iteration with in-situ visualization
    record(fig, "plume.mp4", tstart:Δt:tend-Δt) do tnow
        println(tnow)
        transport(Δx, Δy, Δz, Float64(Dates.value(Dates.Second(Δt))), dH, dV, conc, wind)
        update_plot!(points, colors, conc, 1)
    end

end #let