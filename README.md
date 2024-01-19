# ThesisExperiments

This code base is using the [Julia Language](https://julialang.org/) and
[DrWatson](https://juliadynamics.github.io/DrWatson.jl/stable/)
to make a reproducible scientific project named

> ThesisExperiments

It is authored by Hossein Pourbozorg. the version of this repo that been used in my thesis is tagged as `BaseVersion` and all experiments executed with Julia 1.9.4 and Python 3.9.13.

The main script that can generate the results is named `expr-patchnr.jl`.

To (locally) reproduce this project, do the following:

 1. Download and install Julia. see https://julialang.org/downloads/

 2. Download and install Python. see https://www.python.org/downloads/

 3. Download this code base. Notice that raw data are typically not included in the
    git-history and may need to be downloaded independently.

 4. Download LoDoPaB-CT dataset and put it in `data/lodoct`. see https://zenodo.org/records/3384092

 5. Open a Julia console and do:

    ```
    julia> using Pkg
    julia> Pkg.add("DrWatson") # install globally, for using `quickactivate`
    julia> Pkg.activate("path/to/this/project")
    julia> Pkg.instantiate()
    ```

 6. Open a Shell console and do:

   ```
   python -m pip install -r path/to/this/project/py_reqs.txt
   ```

This will install all necessary tools and packages for you to be able to run the scripts and
everything should work out of the box, including correctly finding local paths.

You may notice that most scripts start with the commands:

```julia
using DrWatson
@quickactivate "ThesisExperiments"
```

which auto-activate the project and enable local path handling from DrWatson.

To run a script do:

```
julia path/to/this/project/scripts/script_name.jl
```

Some documentation has been set up for this project. It can be viewed by
running the file `docs/make.jl`, and then launching the generated file
`docs/build/index.html`.
Alternatively, the documentation may be already hosted online.
If this is the case it should be at:

https://prbzrg.github.io/ThesisExperiments/dev/
