# Build the extracted FSMN-VAD as both:
#   1) fsmn_vad.dll / libfsmn_vad.so  — shared lib exposing the C API (for Python ctypes etc.)
#   2) llama-funasr-vad.exe           — the CLI tool (WAV -> printed segments)
#
# The llama.cpp build toolchain is required (it provides ggml).
#   Windows:  install Visual Studio Build Tools (MSVC) or MinGW-w64 + CMake
#   macOS:    brew install cmake
#   Linux:    apt install cmake g++
#
#   cmake -B build -DCMAKE_BUILD_TYPE=Release
#   cmake --build build -j --config Release
#   -> binaries in build/bin/
cmake_minimum_required(VERSION 3.16)
project(fsmn-vad-extract CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# MinGW: static-link the C/C++/winpthread runtimes so fsmn_vad.dll runs on any
# Windows machine with no bundled libgcc/libstdc++/libwinpthread DLLs.
if(MINGW)
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -static -static-libgcc -static-libstdc++")
endif()

include(FetchContent)
set(LLAMA_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS    OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER   OFF CACHE BOOL "" FORCE)
set(LLAMA_CURL           OFF CACHE BOOL "" FORCE)
FetchContent_Declare(llama
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp.git
    GIT_TAG        8086439a4cea94c71a5dfb8fe4ad1546aebd640f)
FetchContent_MakeAvailable(llama)

find_package(Threads REQUIRED)

# --- shared library with the C API (for Python ctypes / other FFI) ---
add_library(fsmn_vad SHARED funasr_vad_wrapper.cpp)
target_include_directories(fsmn_vad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fsmn_vad PRIVATE ggml Threads::Threads)
set_target_properties(fsmn_vad PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
# Drop the MinGW "lib" prefix so the file is exactly fsmn_vad.dll.
set_target_properties(fsmn_vad PROPERTIES OUTPUT_NAME "fsmn_vad" PREFIX "")

# --- CLI tool ---
add_executable(llama-funasr-vad funasr-vad.cpp)
target_include_directories(llama-funasr-vad PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(llama-funasr-vad PRIVATE ggml Threads::Threads)
install(TARGETS llama-funasr-vad RUNTIME DESTINATION bin)
