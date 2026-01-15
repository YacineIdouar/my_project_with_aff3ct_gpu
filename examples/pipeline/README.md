# How to compile this example

Make sure to have done the instructions from the `README.md` file at the root of 
this repository before doing this. AFF3CT should have been compiled before doing 
the following.

Compile the code on Linux/MacOS/MinGW/WSL:

```bash
mkdir build
cd build
cmake .. -G"Unix Makefiles" -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="-Wall -funroll-loops -march=native" \
         -DAFF3CT_DIR=$(pwd)/../../../aff3ct_install/lib/cmake/aff3ct \
         -Dcpptrace_DIR=$(pwd)/../../../aff3ct_install/lib/cmake/cpptrace
cmake --build . -j 1
```

The source code of this mini project is in `src/main.cpp`.
The compiled binary is in `build/bin/my_project`.

Example of command lines for transferring a file with this example:

```bash
./bin/my_project -K 1023 -N 1023 --src-type USER_BIN --src-no-reset --chn-implem FAST --chn-type AWGN --snk-type USER_BIN --src-path <INPUT_FILE>   --snk-path <OUTPUT_FILE>
./bin/my_project -K 1023 -N 1023 --src-type USER_BIN --src-no-reset --chn-implem FAST --chn-type AWGN --snk-type USER_BIN --src-path CMakeCache.txt --snk-path CMakeCache.rx1.txt
./bin/my_project -K  128 -N 1024 --src-type USER_BIN --src-no-reset --chn-implem FAST --chn-type AWGN --snk-type USER_BIN --src-path CMakeCache.txt --snk-path CMakeCache.rx2.txt
```

> [!warning]
> If you are encountering some problems with 
> [`cpptrace`](https://github.com/jeremy-rifkin/cpptrace), try to compile AFF3CT
> and StreamPU without linking with it (see the `-DSPU_STACKTRACE="OFF"` 
> option). Please refer to the README at the root of this repository.
