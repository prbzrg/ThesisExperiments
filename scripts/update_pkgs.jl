using Pkg

Pkg.precompile(; strict = true)
Pkg.update()
Pkg.precompile(; strict = true)
Pkg.build()
Pkg.precompile(; strict = true)
Pkg.gc()
Pkg.precompile(; strict = true)
