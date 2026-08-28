# Minimal CMake toolchain file for cross-compiling qwen3-tts.cpp (+ GGML)
# to a real iOS/iPadOS arm64 device (not simulator).
#
# Written for issue #1 (iPad Air 5 on-device RTF/memory benchmark). Only the
# OS64 (arm64 device) target is implemented — no simulator, no watchOS/tvOS.
#
# Usage:
#   cmake -B build-ios -G Xcode \
#         -DCMAKE_TOOLCHAIN_FILE=cmake/ios.toolchain.cmake \
#         -DPLATFORM=OS64 \
#         -DDEPLOYMENT_TARGET=17.0 \
#         ...
#
# Non-Xcode generators (e.g. Ninja/Unix Makefiles) also work since we set
# CMAKE_OSX_SYSROOT/ARCHITECTURES/DEPLOYMENT_TARGET directly, which the Apple
# Clang driver picks up regardless of generator.

set(CMAKE_SYSTEM_NAME iOS)
set(CMAKE_SYSTEM_PROCESSOR arm64)

if(NOT DEFINED DEPLOYMENT_TARGET)
    set(DEPLOYMENT_TARGET "17.0")
endif()

# Only OS64 (arm64 physical device) is supported by this minimal toolchain.
set(PLATFORM "OS64")
set(CMAKE_OSX_ARCHITECTURES "arm64")
set(CMAKE_OSX_SYSROOT iphoneos)
set(CMAKE_OSX_DEPLOYMENT_TARGET "${DEPLOYMENT_TARGET}")
set(CMAKE_IOS_DEPLOYMENT_TARGET "${DEPLOYMENT_TARGET}")

execute_process(
    COMMAND xcrun --sdk iphoneos --show-sdk-path
    OUTPUT_VARIABLE CMAKE_OSX_SYSROOT_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
set(CMAKE_OSX_SYSROOT "${CMAKE_OSX_SYSROOT_PATH}")

# Locate the Apple toolchain compilers explicitly (avoids picking up a
# Homebrew/other clang on PATH).
execute_process(COMMAND xcrun -f clang OUTPUT_VARIABLE CMAKE_C_COMPILER OUTPUT_STRIP_TRAILING_WHITESPACE)
execute_process(COMMAND xcrun -f clang++ OUTPUT_VARIABLE CMAKE_CXX_COMPILER OUTPUT_STRIP_TRAILING_WHITESPACE)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -miphoneos-version-min=${DEPLOYMENT_TARGET} -arch arm64 -isysroot ${CMAKE_OSX_SYSROOT}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -miphoneos-version-min=${DEPLOYMENT_TARGET} -arch arm64 -isysroot ${CMAKE_OSX_SYSROOT}")
set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -arch arm64 -isysroot ${CMAKE_OSX_SYSROOT}")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -arch arm64 -isysroot ${CMAKE_OSX_SYSROOT}")

# Never try to run target-architecture binaries on the host during configure
# (e.g. try_run checks) — cross-compiling.
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# Only look for headers/libraries in the SDK, never on the host filesystem.
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
