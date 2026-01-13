# How to compile this example

Make sure to have done the instructions from the `README.md` file at the root of 
this repository before doing this. AFF3CT should have been compiled before doing 
the following.

Compile the code on Linux/MacOS/MinGW/WSL:

```bash
mkdir build
cd build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-funroll-loops -march=native" \
         -DAFF3CT_DIR=$(pwd)/../../../aff3ct_install/lib/cmake/aff3ct \
         -Dcpptrace_DIR=$(pwd)/../../../aff3ct_install/lib/cmake/cpptrace
cmake --build . -j 1
```

The source code of this mini project is in `src/main.cpp`.
The compiled binary is in `build/bin/my_project`.

> [!warning]
> If you are encountering some problems with 
> [`cpptrace`](https://github.com/jeremy-rifkin/cpptrace), try to compile AFF3CT
> and StreamPU without linking with it (see the `-DSPU_STACKTRACE="OFF"` 
> option). Please refer to the README at the root of this repository.
