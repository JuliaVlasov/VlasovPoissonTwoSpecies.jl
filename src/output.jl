struct OutputManager

     data :: Data 
     mesh_x :: Mesh
     mesh_v :: Mesh 
     fe_eq_init :: Matrix{Float64}
     fi_eq_init :: Matrix{Float64}
     nb_outputs :: Int
     t :: Vector{Float64}
     compute_energy_eq :: Bool
     energy_fe :: Vector{Float64}
     energy_fi :: Vector{Float64}

     function OutputManager(data, mesh_x, mesh_v, fe_eq_init, fi_eq_init)
         nb_outputs = 0
         t = [0.0]
         compute_energy_eq = false
         self.energy_fe = [compute_energy(mesh_x, mesh_v, fe_eq_init)]
         self.energy_fi = [compute_energy(mesh_x, mesh_v, fi_eq_init)]

         new(data, mesh_x, mesh_v, fe_eq_init, fi_eq_init, nb_outputs, t,
             compute_energy_eq, energy_fe, energy_fi)

     end

end

function save(self, fe, fi, fe_eq, fi_eq, t)

    data   = self.data
    mesh_x = self.mesh_x
    mesh_v = self.mesh_v

    push!(self.t, t)
    e_fe, e_fi = compute_normalized_energy(fe, fi)
    push!(self.energy_fe, e_fe)
    push!(self.energy_fi, e_fi)

end

function compute_normalized_energy(self, fe, fi)

    mesh_x = self.mesh_x
    mesh_v = self.mesh_v
    
    return (get_normalized_energy(mesh_x, mesh_v, fe, self.energy_fe_eq_init),
            get_normalized_energy(mesh_x, mesh_v, fi, self.energy_fi_eq_init))

end

"""
    Return the energy e_f = int v^2 f dv
"""
function compute_energy(mesh_x, mesh_v, f)

    v  = mesh_v.x
    dx = mesh_x.dx
    dv = mesh_v.dx
    energy = f .* v.^2 

    return dv * dx * sum(energy)

end


"""
    Return the normalized energy |e_f - e_eq| / e_eq
    where e_f is the energy of f.
"""
function get_normalized_energy(mesh_x, mesh_v, f, energy_eq)
    return abs(compute_energy(mesh_x, mesh_v, f) - energy_eq) / energy_eq
end


