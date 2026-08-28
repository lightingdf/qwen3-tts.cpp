cmake_minimum_required(VERSION 3.14)
project(qwen3-tts-ggml VERSION 0.1.0 LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Static libs are linked into the qwen3tts shared lib (C API), so they must be PIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
    # -march=native requires the compiler to probe the *host* CPU, which is
    # meaningless (and fails outright) when cross-compiling for iOS/tvOS/etc.
    # Only apply it for native (non-cross-compiling) builds, i.e. plain macOS.
    if(NOT CMAKE_SYSTEM_NAME MATCHES "iOS|tvOS|visionOS|watchOS")
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
    else()
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
    endif()
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
endif()

# Timing instrumentation option
option(QWEN3_TTS_TIMING "Enable detailed timing instrumentation" OFF)
if(QWEN3_TTS_TIMING)
    add_compile_definitions(QWEN3_TTS_TIMING)
endif()

# CoreML code predictor bridge (macOS only)
option(QWEN3_TTS_COREML "Enable CoreML code predictor bridge on macOS" ON)
if(APPLE AND QWEN3_TTS_COREML)
    enable_language(OBJCXX)
endif()

if(WIN32)
    add_compile_definitions(_USE_MATH_DEFINES NOMINMAX)
endif()

# Find threads
find_package(Threads REQUIRED)

# GGML library paths
set(GGML_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ggml")
# Allow pointing at an alternate pre-built GGML tree (e.g. a cross-compiled
# ggml/build-ios) via -DGGML_BUILD_DIR=... on the cmake command line.
if(NOT DEFINED GGML_BUILD_DIR)
    set(GGML_BUILD_DIR "${GGML_DIR}/build")
endif()

# Detect Metal backend availability
if(APPLE AND EXISTS "${GGML_BUILD_DIR}/src/ggml-metal/libggml-metal.dylib")
    set(GGML_HAS_METAL ON)
    message(STATUS "GGML Metal backend found — GPU acceleration enabled")
else()
    set(GGML_HAS_METAL OFF)
endif()

# Text tokenizer library
add_library(text_tokenizer STATIC
    src/text_tokenizer.cpp
    src/gguf_loader.cpp
)
target_include_directories(text_tokenizer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(text_tokenizer PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(text_tokenizer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(text_tokenizer PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()

# TTS Transformer library (GGML-based + optional CoreML bridge)
set(TTS_TRANSFORMER_SOURCES
    src/gguf_loader.cpp
    src/tts_transformer.cpp
)
if(APPLE AND QWEN3_TTS_COREML)
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor.mm)
    set_source_files_properties(src/coreml_code_predictor.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
else()
    list(APPEND TTS_TRANSFORMER_SOURCES src/coreml_code_predictor_stub.cpp)
endif()

add_library(tts_transformer STATIC
    ${TTS_TRANSFORMER_SOURCES}
)
target_include_directories(tts_transformer PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(tts_transformer PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(tts_transformer PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(tts_transformer PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()
if(APPLE AND QWEN3_TTS_COREML)
    target_link_libraries(tts_transformer PUBLIC
        "-framework Foundation"
        "-framework CoreML"
    )
endif()

# Audio tokenizer encoder library (GGML-based)
add_library(audio_tokenizer_encoder STATIC
    src/audio_tokenizer_encoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_encoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_encoder PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(audio_tokenizer_encoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(audio_tokenizer_encoder PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()

# Audio tokenizer decoder library (GGML-based)
add_library(audio_tokenizer_decoder STATIC
    src/audio_tokenizer_decoder.cpp
    src/gguf_loader.cpp
)
target_include_directories(audio_tokenizer_decoder PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_directories(audio_tokenizer_decoder PUBLIC
    ${GGML_BUILD_DIR}/src
    ${GGML_BUILD_DIR}/src/ggml-metal
)
target_link_libraries(audio_tokenizer_decoder PUBLIC
    ggml
    ggml-base
    ggml-cpu
    Threads::Threads
)
if(GGML_HAS_METAL)
    target_link_libraries(audio_tokenizer_decoder PUBLIC ggml-metal "-framework Metal" "-framework MetalKit")
endif()

# Qwen3 TTS library (full pipeline)
add_library(qwen3_tts STATIC
    src/qwen3_tts.cpp
)
target_include_directories(qwen3_tts PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${GGML_DIR}/include
)
target_link_libraries(qwen3_tts PUBLIC
    text_tokenizer
    tts_transformer
    audio_tokenizer_encoder
    audio_tokenizer_decoder
    Threads::Threads
)
if(WIN32)
    target_link_libraries(qwen3_tts PUBLIC psapi)
endif()

# Static library with C API (same source as qwen3tts_shared below, built as
# a static archive too). Exists for embedders that can't/don't want to carry
# a shared library/dylib around — e.g. an iOS app target linking everything
# statically, such as VoiceEditor issue #1's on-device benchmark harness.
add_library(qwen3tts_c_api STATIC
    src/qwen3tts_c_api.cpp
)
target_include_directories(qwen3tts_c_api PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_c_api PUBLIC
    qwen3_tts
)
if(APPLE)
    # See the identical note on qwen3tts_shared below: qwen3tts_c_api.cpp
    # calls the Objective-C runtime directly without #import-ing anything
    # that would otherwise pull libobjc in as a link dependency.
    target_link_libraries(qwen3tts_c_api PRIVATE objc)
endif()

# Shared library with C API (for Nim FFI)
add_library(qwen3tts_shared SHARED
    src/qwen3tts_c_api.cpp
)
set_target_properties(qwen3tts_shared PROPERTIES
    OUTPUT_NAME "qwen3tts"
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)
target_include_directories(qwen3tts_shared PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
)
target_link_libraries(qwen3tts_shared PRIVATE
    qwen3_tts
)
if(APPLE)
    # qwen3tts_c_api.cpp calls the Objective-C runtime directly (objc_msgSend
    # etc.) to drive an NSAutoreleasePool around synthesis calls, without
    # #import-ing anything that would otherwise pull in libobjc as a link
    # dependency. On macOS this got linked in transitively via the CoreML
    # (.mm) bridge target; with QWEN3_TTS_COREML=OFF (e.g. the iOS build)
    # nothing else references libobjc and the link fails with undefined
    # _objc_msgSend/_objc_getClass/_sel_registerName symbols.
    target_link_libraries(qwen3tts_shared PRIVATE objc)
endif()

# CLI executable
add_executable(qwen3-tts-cli
    src/main.cpp
)
target_link_libraries(qwen3-tts-cli PRIVATE
    qwen3_tts
)

# Test executable for tokenizer
add_executable(test_tokenizer
    tests/test_tokenizer.cpp
)
target_link_libraries(test_tokenizer PRIVATE
    text_tokenizer
    Threads::Threads
)

# Test executable for audio encoder
add_executable(test_encoder
    tests/test_encoder.cpp
)
target_link_libraries(test_encoder PRIVATE
    audio_tokenizer_encoder
    Threads::Threads
)

# Test executable for transformer
add_executable(test_transformer
    tests/test_transformer.cpp
)
target_link_libraries(test_transformer PRIVATE
    tts_transformer
    Threads::Threads
)

# Test executable for audio decoder
add_executable(test_decoder
    tests/test_decoder.cpp
)
target_link_libraries(test_decoder PRIVATE
    audio_tokenizer_decoder
    Threads::Threads
)

# Install targets
install(TARGETS text_tokenizer tts_transformer audio_tokenizer_encoder audio_tokenizer_decoder qwen3_tts qwen3tts_c_api qwen3tts_shared qwen3-tts-cli
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)
install(FILES
    src/gguf_loader.h
    src/text_tokenizer.h
    src/tts_transformer.h
    src/coreml_code_predictor.h
    src/audio_tokenizer_encoder.h
    src/audio_tokenizer_decoder.h
    src/qwen3_tts.h
    src/qwen3tts_c_api.h
    DESTINATION include
)

# CTest support
enable_testing()
add_test(NAME tokenizer_test
    COMMAND test_tokenizer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME encoder_test
    COMMAND test_encoder --tokenizer models/qwen3-tts-0.6b-f16.gguf --audio clone.wav --reference reference/ref_audio_embedding.bin
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME transformer_test
    COMMAND test_transformer
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_test(NAME decoder_test
    COMMAND test_decoder
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
