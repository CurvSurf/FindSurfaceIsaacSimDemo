cmake_minimum_required(VERSION 3.16)
project(FindSurfaceIsaacSimPicking LANGUAGES CXX)

add_library(FindSurfaceIsaacSimPicking SHARED FindSurfacePicking.cpp)
target_compile_features(FindSurfaceIsaacSimPicking PRIVATE cxx_std_17)
set_target_properties(FindSurfaceIsaacSimPicking PROPERTIES
    CXX_EXTENSIONS OFF
    OUTPUT_NAME "FindSurfaceIsaacSimPicking"
)
if(WIN32)
    set_target_properties(FindSurfaceIsaacSimPicking PROPERTIES PREFIX "")
endif()

if(MSVC)
    target_compile_options(FindSurfaceIsaacSimPicking PRIVATE /O2 /fp:precise)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(FindSurfaceIsaacSimPicking PRIVATE -O3)
    if(WIN32 AND MINGW)
        target_link_options(FindSurfaceIsaacSimPicking PRIVATE -static -static-libgcc -static-libstdc++)
    endif()
endif()

if(NOT WIN32)
    set_target_properties(FindSurfaceIsaacSimPicking PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
        POSITION_INDEPENDENT_CODE ON
    )
endif()
