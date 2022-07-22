export Mesh

struct Mesh

    x     :: Vector{Float64}
	x_min :: Float64
	x_max :: Float64
	nx    :: Int
	dx    :: Float64

	function Mesh( x_min, x_max, nx)

        x = LinRange(x_min, x_max, nx+1)[1:end-1]
        dx = x[2] - x[1]
        new( x, x_min, x_max, nx, dx)

    end

end
