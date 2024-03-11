using Conda

Conda.pip_interop(true)

Conda.update()

Conda.add_channel("pytorch")
Conda.add_channel("odlgroup")

Conda.add("numpy<1.24")
Conda.add("scikit-image")
Conda.add("pytorch")
Conda.add("odl")
# Conda.add("dival")

Conda.pip("install", "git+https://github.com/jleuschn/dival.git")
# Conda.pip("install", "git+https://github.com/odlgroup/odl.git")
Conda.pip("install", "https://github.com/odlgroup/odl/archive/refs/heads/master.zip")

Conda.clean()
