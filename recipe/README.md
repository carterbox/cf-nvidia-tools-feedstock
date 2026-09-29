# cf-nvidia-tools

This package contains CLI tools for validating and linting NVIDIA's conda recipes on
conda-forge. The tools are hosted directly in this feedstock; there is no external source
code repository for these tools at this time.

## API policy

This project will use semantic versioning, and will not have a beta release. In other words,
it should be safe to pin this tool with an expression like `>=1.0,<2`. Backward incompatible
changes for any tool in this package will be accompanied by a major version bump. New
features will increase the minor version. Anything considered a bug fix will be a patch
release.

## check-glibc

### When to use this tool

The package is binary redistrbution which links to glibc.

### Why is this tool needed

For NVIDIA conda packages which are redists (binary redistributions) there is no compiling
when building the package, so we do not automatically know if the glibc provided by `{{
stdlib('c') }}` is new enough for the binaries in the package. If we don't set a new enough
version for `c_stdlib_version`, end users can get undefined symbol errors. This tool is used
for checking that a binary redist has correctly specified the `c_stdlib_version` variable in
the recipe.

### How to Use

Use the tool in the build script after installing the binaries to `$PREFIX`. For example:

```bash
check-glibc $PREFIX/lib/libfoo*.so.*
```

Here we have used bash to expand a glob expression and provide `check-glibc` with a list of
files to search for glibc symbols. You can check arbitrary files by setting your own glob
expression or explicitly listing the files. We have narrowed the glob to versioned shared
libraries only because we don't need to check links to shared libraries (that would be
duplicate effort).

> [!CAUTION]
> check-glibc does not expand glob expressions. You must leave the glob expression unquoted
> in order for bash to expand the expression before check-glibc is called.

> [!CAUTION]
> check-glibc will ignore links. In other words, you must name the *actual files* not
> symbolic links.

The tool will exit with status 1 if `c_stdlib_version` is older than newest symbol detected
in any of the binaries. The newest symbol detected for each binary is logged to the
terminal.

If this check fails, increase `c_stdlib_version` to a version newer than the version
detected by this tool.

## check-cuda-arch

### When to use this tool

The package is a binary with device code.

### Why is this tool needed

For NVIDIA conda packages which are redists, there is no compiling when building the
package, so we do not automatically know whether the binary was build to support the lowest
CUDA architecture supported by CUDA toolkit (sm_50 for CUDA 12 and sm_75 for CUDA 13). If we
don't use the new `cuda-arch` package to set a minimum supported CUDA architecture when the
binary does not support the minimum arch for a CUDA major version, users can experience
segmentation faults when conda installs a package that does not support their system. This
reports whether `cuda_arch_version` matches the computed lowest common CUDA architecture
across a group of binaries i.e. the lowest architecture that every binary in the group still
supports.

### How to Use

Use the tool in the build script after installing the binaries to `$PREFIX`, with
`cuda_arch_version` set in the environment. For example:

```bash
cuda_arch_version=7.5 check-cuda-arch $PREFIX/lib/libfoo*.so.*
```

The check exits 0 on a match, 1 on a mismatch or an unreadable binary, and 2 on a bad
invocation.

## trim-cuda-archs

### When to use this tool

The recipe builds from source with `nvcc` and needs to restrict the CUDA architectures it
targets to those at or above `cuda_arch_version`, using one of `CUDAARCHS`,
`CF_TORCH_CUDA_ARCH_LIST`, or `NVCC_GENCODE`.

### Why is this tool needed

By default, these variables are set to a wide range of CUDA architectures, but an upstream
project may not support all possible CUDA architectures (especially machine learning
projects which are optimized for only a few datacenter class devices). This tool trims one
of the three variables down to the architectures at or above a given minimum, without
requiring the recipe to hand-parse each variable's own syntax. In this way, recipe
maintainers can still use these variables (which will be updated to track new hardware
releases) while correctly specifying the minimum architecture requirements.

### How to Use

A subprocess cannot modify its parent shell's environment directly, so capture the tool's
stdout and reassign it to the variable yourself:

```bash
export CUDAARCHS="$(trim-cuda-archs CUDAARCHS 7.5)"
```

```powershell
$env:CUDAARCHS = trim-cuda-archs CUDAARCHS 7.5
```

```bat
for /f "delims=" %i in ('trim-cuda-archs CUDAARCHS 7.5') do set CUDAARCHS=%i
```

If the named variable is unset or empty, nothing is printed and the assignment becomes
empty. A report of which architectures were dropped is printed to stderr.
