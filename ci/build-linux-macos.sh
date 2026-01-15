#!/bin/bash
set -x

if [ -z "$EXAMPLES" ]
then
	echo "Please define the 'EXAMPLES' environment variable."
	exit 1
fi

if [ -z "$CXX" ]
then
	echo "Please define the 'CXX' environment variable."
	exit 1
fi

if [ -z "$AFF3CT_GIT_VERSION" ]
then
	echo "Please define the 'AFF3CT_GIT_VERSION' environment variable."
	exit 1
fi

if [ -z "$BUILD" ]
then
	echo "The 'BUILD' environment variable is not set, default value = 'build_linux_macos'."
	BUILD="build_linux_macos"
fi

if [ -z "$THREADS" ]
then
	echo "The 'THREADS' environment variable is not set, default value = 1."
	THREADS=1
fi

if [[ $CXX == icpc ]]; then
	source /opt/intel/vars-intel.sh
fi

# Compile the AFF3CT library
cd lib/aff3ct
mkdir $BUILD
cd $BUILD
cmake .. -G"Unix Makefiles" -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="$CFLAGS" \
         -DCMAKE_EXE_LINKER_FLAGS="$LFLAGS" -DAFF3CT_COMPILE_EXE="OFF" -DAFF3CT_COMPILE_STATIC_LIB="ON"
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
cmake --build . -j $THREADS
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
cmake --install . --prefix ../../../aff3ct_install
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
ln -sfn ../../../aff3ct_install/lib/cmake/aff3ct-* ../../../aff3ct_install/lib/cmake/aff3ct
rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
cd ..

# Compile all the projects using AFF3CT
cd ../../examples
for example in ${EXAMPLES[*]}; do
	cd $example
	mkdir $BUILD
	cd $BUILD
	cmake .. -G"Unix Makefiles" -DCMAKE_CXX_COMPILER=$CXX -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CXX_FLAGS="$CFLAGS" \
	         -DCMAKE_EXE_LINKER_FLAGS="$LFLAGS" -DAFF3CT_DIR=$(pwd)/../../../aff3ct_install/lib/cmake/aff3ct \
             -Dcpptrace_DIR=$(pwd)/../../../aff3ct_install/lib/cmake/cpptrace
	rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
	cmake --build . -j $THREADS
	rc=$?; if [[ $rc != 0 ]]; then exit $rc; fi
	cd ../..
done