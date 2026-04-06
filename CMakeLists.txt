cmake_minimum_required(VERSION 3.14)
project(EGBI
    VERSION 1.0.0
    DESCRIPTION "Entropy-Gated Bit-Vector Indexing - High-speed fuzzy matching library"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# --- Header-only library target ---
add_library(egbi INTERFACE)
target_include_directories(egbi INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Platform-specific compiler flags
if(MSVC)
    target_compile_options(egbi INTERFACE /W4)
else()
    target_compile_options(egbi INTERFACE -Wall -Wextra -O2)
    # Enable hardware popcount on x86
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-mpopcnt" HAS_POPCNT)
    if(HAS_POPCNT)
        target_compile_options(egbi INTERFACE -mpopcnt)
    endif()
endif()

# --- Options ---
option(EGBI_BUILD_CLI      "Build EGBI-CLI tool"    ON)
option(EGBI_BUILD_EXAMPLES "Build example programs" ON)
option(EGBI_BUILD_TESTS    "Build test programs"    ON)

# --- CLI tool ---
if(EGBI_BUILD_CLI)
    add_executable(egbi-cli cli/main.cpp)
    target_link_libraries(egbi-cli PRIVATE egbi)
endif()

# --- Examples ---
if(EGBI_BUILD_EXAMPLES)
    add_executable(egbi_example examples/basic_usage.cpp)
    target_link_libraries(egbi_example PRIVATE egbi)
endif()

# --- Tests ---
if(EGBI_BUILD_TESTS)
    enable_testing()
    add_executable(egbi_test tests/test_egbi.cpp)
    target_link_libraries(egbi_test PRIVATE egbi)
    add_test(NAME egbi_test COMMAND egbi_test)
endif()

# --- Install ---
include(GNUInstallDirs)
install(TARGETS egbi EXPORT egbi-targets)
install(DIRECTORY include/egbi DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(EXPORT egbi-targets
    FILE egbi-config.cmake
    NAMESPACE egbi::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/egbi
)
