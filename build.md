# Building StreamPU (TidePU), AFF3CT and the pipeline project

The project (`examples/pipeline`) is built on top of two libraries, which must be
built and installed in this order:

1. **StreamPU / TidePU**, the runtime, installed in `lib/spu_install`;
2. **AFF3CT**, built against that StreamPU and installed in `lib/aff3ct_install`;
3. **the project**, built against both.

All the commands below start from the repository root:

```bash
export ROOT=$(pwd)   # the root of my_project_with_aff3ct_gpu
```

## 0. Prerequisites

- CMake ≥ 3.18 and a C++17 compiler.
- One toolchain per GPU backend you want to use:

| Backend | Needs |
| --- | --- |
| CUDA | CUDA toolkit (`nvcc`) |
| Vulkan | Vulkan loader and headers, plus `glslc` to compile the shaders |
| SYCL (oneAPI) | Intel LLVM `clang++` with `-fsycl` (here `/opt/llvm/build/bin/clang++`) |
| HIP | ROCm |

- Optional: `hwloc` (thread pinning in StreamPU, `-DSPU_LINK_HWLOC=ON`).

Fetch the submodules. StreamPU is itself a submodule of AFF3CT, in `lib/aff3ct/lib/streampu`:

```bash
git submodule update --init --recursive
```

### GPU architecture

CUDA needs the compute capability of the GPU, without the dot:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv,noheader   # 8.9 -> 89
```

HIP needs the `gfx` name of the GPU (e.g. `gfx90a`). Use the same value for
StreamPU and for the project.

## 1. StreamPU / TidePU

The batched modules (`--chn-api CUDA_BATCHED`, `--dec-api CUDA_BATCHED`) are written
with StreamPU's Generic module. Check out a TidePU branch that has it:
`tidepu_generic_module`, or any branch it has been merged into.

```bash
cd $ROOT/lib/aff3ct/lib/streampu
git fetch origin
git checkout tidepu_generic_module
git submodule update --init --recursive        # rang, cpptrace, json

cmake -S . -B build-install \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/opt/llvm/build/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native" \
  -DSPU_COMPILE_SHARED_LIB=ON -DSPU_COMPILE_STATIC_LIB=ON \
  -DSPU_STACKTRACE=ON -DSPU_TESTS=OFF \
  -DSPU_ENABLE_CUDA=ON -DCUDA_ARCH=89 \
  -DSPU_ENABLE_VULKAN=ON \
  -DSPU_ENABLE_OAPI_SYCL=ON -DBACKEND="spir64,nvptx64-nvidia-cuda" \
  -DCMAKE_INSTALL_PREFIX=$ROOT/lib/spu_install
cmake --build build-install -j 16
cmake --install build-install
```

Enable only the backends you have. Backend options:

| Backend | StreamPU options |
| --- | --- |
| CUDA | `-DSPU_ENABLE_CUDA=ON -DCUDA_ARCH=<sm>` |
| Vulkan | `-DSPU_ENABLE_VULKAN=ON` |
| SYCL (oneAPI) | `-DSPU_ENABLE_OAPI_SYCL=ON -DBACKEND=<targets>` (needs the Intel LLVM `clang++`) |
| HIP | `-DSPU_ENABLE_HIP=ON -DHIP_ARCH=<gfx...>` |

For a CUDA-only build, any C++17 compiler works and the Intel LLVM one is not needed.

The installation also provides cpptrace (`lib/spu_install/lib/cmake/cpptrace`), which
AFF3CT and the project use.

## 2. AFF3CT

AFF3CT embeds its own copy of StreamPU by default. `-DAFF3CT_INCLUDE_SPU_LIB=OFF` makes
it use the StreamPU installed above instead. Without this option, the project ends up
with two different StreamPUs.

```bash
cd $ROOT/lib/aff3ct
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native" \
  -DAFF3CT_COMPILE_EXE=OFF -DAFF3CT_COMPILE_STATIC_LIB=ON -DAFF3CT_COMPILE_SHARED_LIB=ON \
  -DAFF3CT_INCLUDE_SPU_LIB=OFF \
  -DStreamPU_DIR=$ROOT/lib/spu_install/lib/cmake/streampu \
  -Dcpptrace_DIR=$ROOT/lib/spu_install/lib/cmake/cpptrace
cmake --build build -j 16
cmake --install build --prefix $ROOT/lib/aff3ct_install
```

The installation puts its CMake files in a versioned folder
(`lib/cmake/aff3ct-<version>`). The project looks for `lib/cmake/aff3ct`, so make
`aff3ct` a link to the version you just installed:

```bash
cd $ROOT/lib/aff3ct_install/lib/cmake
ls -d aff3ct-*                                  # e.g. aff3ct-4.5.0-3-g2ae5df6
rm -rf aff3ct && ln -s aff3ct-4.5.0-3-g2ae5df6 aff3ct
```

> [!warning]
> If `aff3ct` is a real directory rather than a link, `ln -sfn` creates the link
> **inside** it, and the project silently keeps using the older AFF3CT. Hence the
> `rm -rf` above. Check with `ls -l`: `aff3ct` must show as `aff3ct -> aff3ct-<version>`.

## 3. The project (`examples/pipeline`)

```bash
cd $ROOT/examples/pipeline
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_CXX_COMPILER=/opt/llvm/build/bin/clang++ \
  -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native" \
  -DAFF3CT_DIR=$ROOT/lib/aff3ct_install/lib/cmake/aff3ct \
  -DStreamPU_DIR=$ROOT/lib/spu_install/lib/cmake/streampu \
  -Dcpptrace_DIR=$ROOT/lib/spu_install/lib/cmake/cpptrace \
  -DDECODER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DDECODER_VULKAN=ON \
  -DDECODER_SYCL=ON -DSYCL_COMPILER=oneapi -DSYCL_BACKEND="spir64,nvptx64-nvidia-cuda"
cmake --build build -j 16
```

Backend options:

| Backend | Project options |
| --- | --- |
| CUDA | `-DDECODER_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=<sm>` |
| Vulkan | `-DDECODER_VULKAN=ON` |
| SYCL (oneAPI) | `-DDECODER_SYCL=ON -DSYCL_COMPILER=oneapi -DSYCL_BACKEND=<targets>` |
| SYCL (AdaptiveCpp) | `-DDECODER_SYCL=ON -DSYCL_COMPILER=acpp` |
| HIP | `-DDECODER_HIP=ON -DHIP_ARCH=<gfx...>` |

Two rules:

- **Only enable backends that StreamPU was built with.**
- **If StreamPU was built with Vulkan, the project needs `-DDECODER_VULKAN=ON`.**
  StreamPU's CMake package links against `Vulkan::Vulkan` but does not look for Vulkan
  itself. The project only looks for it when `DECODER_VULKAN` is on, so without it
  the configuration fails (see [Troubleshooting](#troubleshooting)).

Don't put `-fsycl` in `CMAKE_CXX_FLAGS`: the project removes it and only applies it to
the SYCL sources.

The executables are written to `build/bin/`:

| Executable | Source | Notes |
| --- | --- | --- |
| `pipeline_cpu` | `src/main.cpp` | |
| `pipeline_gpu` | `src/main_gpu.cpp` | takes the GPU options of section 4 |
| `pipeline_6st` | `src/main_6st.cpp` | same as `pipeline_gpu`, with the channel in a stage of its own (6 stages, so `--pip-threads` takes 6 counts) |
| `pipeline_tidepu_check` | `src/main_tidepu_check.cpp` | checks the batched CUDA channel and decoder against the original CUDA modules; only built with `DECODER_CUDA` |

## 4. Running

AFF3CT looks for the files that the command line names by relative paths
(matrices, `conf/…`) under `AFF3CT_CLI_PATH`:

```bash
export AFF3CT_CLI_PATH=$ROOT/lib/aff3ct
cd $ROOT/examples/pipeline/build
```

Sanity check (arguments: `[K] [N] [n_ite]`, default `1000 2000 10`). It must end with
`# All the checks passed.`:

```bash
./bin/pipeline_tidepu_check
```

A simulation with the batched CUDA channel and decoder, 8 frames per firing:

```bash
./bin/pipeline_gpu -K 4000 -N 8000 --enc-type LDPC_5G --dec-type BP_FLOODING --dec-implem OMS --dec-off 0.5 \
                   --snr-min 2.5 --snr-max 2.51 -e 100000 -n 4000 \
                   --chn-api CUDA_BATCHED --dec-api CUDA_BATCHED --n-frames 8
```

The GPU options (`--help` lists them all):

| Option | Meaning |
| --- | --- |
| `--n-frames <int>` | Frames per firing, for every module (default 1). `CUDA_BATCHED` modules process them in one launch; the others call their kernel once per frame. |
| `--chn-api <api>` | Channel implementation: `CPU`, `CUDA`, `CUDA_BATCHED`, `CUDA_PRNG`, `HIP_PRNG`, `SYCL_PRNG`, `VULKAN_PRNG`. |
| `--dec-api <api>` | Decoder backend, among those compiled in (`CUDA`, `CUDA_BATCHED`, `HIP`, `SYCL`, `VULKAN`). |
| `--dev-id`, `--plt-id` | GPU device id, and platform id (SYCL only). |
| `--pip-threads <list>` | Threads per pipeline stage, e.g. `1,4,1,1,1` (`pipeline_gpu`) or `1,4,1,1,1,1` (`pipeline_6st`). |

## When to rebuild what

AFF3CT's modules inherit from StreamPU classes (`Module`, `Stateful`, `Task`), and
AFF3CT is compiled against StreamPU's headers. So:

| You changed… | Rebuild |
| --- | --- |
| StreamPU sources only (`src/`) | reinstall StreamPU, then rebuild the project (relink) |
| StreamPU headers (`include/`), or a StreamPU option such as `SPU_STACKTRACE` | reinstall StreamPU, **rebuild and reinstall AFF3CT**, then rebuild the project |
| AFF3CT | reinstall AFF3CT, then rebuild the project |
| the project | the project only |

When in doubt, rebuild all three in order.

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `The link interface of target "spu::spu-shared-lib" contains: Vulkan::Vulkan but the target was not found` | StreamPU was built with Vulkan but the project is configured without it: add `-DDECODER_VULKAN=ON`, or rebuild StreamPU with `-DSPU_ENABLE_VULKAN=OFF`. |
| `Architecture version {CUDA_ARCH} is needed for the CUDA` | StreamPU: add `-DCUDA_ARCH=<sm>`. |
| `Architecture version {HIP_ARCH} is needed for the HIP` | StreamPU: add `-DHIP_ARCH=<gfx...>`. |
| `A backend is needed for the OneAPI version` | StreamPU: add `-DBACKEND=<targets>`. |
| `SYCL_BACKEND is needed for the oneAPI SYCL_COMPILER` | Project: add `-DSYCL_BACKEND=<targets>`. |
| `Could not find a package configuration file provided by "StreamPU"` | Wrong `StreamPU_DIR`, or StreamPU was not installed. |
| AFF3CT: target `spu::spu-shared-lib` not found | StreamPU was built without `-DSPU_COMPILE_SHARED_LIB=ON`; AFF3CT's shared library links against StreamPU's shared one. |
| The `*_batched` files don't compile (`Generic` unknown in `spu::module`) | The installed StreamPU has no Generic module: check out a TidePU branch that has it (step 1). |
| Crash in a destructor at startup or at exit, heap corruption (`corrupted size vs. prev_size`, `free(): invalid pointer`) | AFF3CT is out of date with respect to StreamPU's headers or options (e.g. a different `SPU_STACKTRACE`): rebuild and reinstall AFF3CT, then rebuild the project. |
| `'filename' couldn't be opened` | `AFF3CT_CLI_PATH` is not set (section 4). |
| The project uses an older AFF3CT than the one you just installed | `lib/aff3ct_install/lib/cmake/aff3ct` is a real directory instead of a link (end of step 2). |
