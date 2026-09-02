cmake_minimum_required( VERSION 3.23 )

project( rad2 LANGUAGES CXX )

set( CMAKE_CXX_STANDARD 20 )
set( CMAKE_CXX_STANDARD_REQUIRED ON )
set( CMAKE_CXX_EXTENSIONS OFF )

option( RAD2_WERROR "treat warnings as errors" ON )

if ( WIN32 )
    set( CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL" )
    add_compile_definitions( $<$<CONFIG:Debug>:_ITERATOR_DEBUG_LEVEL=0> )
    set( embree_archive embree-4.4.1.sycl.x64.windows.zip )
    set( embree_sha256 fd4a499f3a0a5b3cbe7a9f48134249bdc983ab4029f04592fc1054cace36cf63 )
else()
    set( embree_archive embree-4.4.1.sycl.x86_64.linux.tar.gz )
    set( embree_sha256 bc67b75c3d6b789a13b9cf82c671699b4b6feb1549e7729fc38ebe50bde4951b )
endif()

include( FetchContent )

FetchContent_Declare( embree
    URL https://github.com/RenderKit/embree/releases/download/v4.4.1/${embree_archive}
    URL_HASH SHA256=${embree_sha256}
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)

FetchContent_Declare( glm
    GIT_REPOSITORY https://github.com/g-truc/glm.git
    GIT_TAG 0af55ccecd98d4e5a8d1fad7de25ba429d60e863 # 1.0.1
    SYSTEM
)

set( GLM_BUILD_TESTS OFF CACHE BOOL "" FORCE )
set( BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE )

FetchContent_MakeAvailable( embree glm )

find_package( embree 4.4 REQUIRED CONFIG PATHS "${embree_SOURCE_DIR}" NO_DEFAULT_PATH
    PATH_SUFFIXES lib/cmake/embree-4.4.1 lib64/cmake/embree-4.4.1 )

find_package( TBB REQUIRED CONFIG )

set_target_properties( TBB::tbb PROPERTIES MAP_IMPORTED_CONFIG_DEBUG Release )

add_executable( rad2
    src/bsp.cpp
    src/entities.cpp
    src/filter.cpp
    src/geometry.cpp
    src/lights.cpp
    src/luxels.cpp
    src/log.cpp
    src/main.cpp
    src/patches.cpp
    src/solve.cpp
    src/visibility.cpp
)

set_source_files_properties( src/solve.cpp PROPERTIES
    COMPILE_OPTIONS "-fsycl;-Xclang;-fsycl-allow-func-ptr"
)
target_link_options( rad2 PRIVATE -fsycl )
target_link_libraries( rad2 PRIVATE embree_sycl )

target_link_libraries( rad2 PRIVATE embree TBB::tbb glm::glm )

target_compile_definitions( rad2 PRIVATE
    $<$<PLATFORM_ID:Windows>:NOMINMAX WIN32_LEAN_AND_MEAN _CRT_SECURE_NO_WARNINGS>
)

target_compile_options( rad2 PRIVATE
    -Wall -Wextra -Wshadow
    $<$<BOOL:${RAD2_WERROR}>:-Werror>
    $<IF:$<STREQUAL:${CMAKE_CXX_COMPILER_FRONTEND_VARIANT},MSVC>,/fp:precise,-ffp-model=precise>
)

if ( WIN32 )
    set( rad2_runtime_dlls embree4.dll tbbmalloc.dll sycl8.dll ur_loader.dll
        ur_adapter_level_zero.dll ur_win_proxy_loader.dll )
    list( TRANSFORM rad2_runtime_dlls PREPEND "${embree_SOURCE_DIR}/bin/" )
    list( APPEND rad2_runtime_dlls "$<TARGET_FILE:TBB::tbb>" )

    get_filename_component( rad2_compiler_bin "${CMAKE_CXX_COMPILER}" DIRECTORY )

    foreach( intel_dll libmmd.dll svml_dispmd.dll libircmd.dll )
        if ( EXISTS "${rad2_compiler_bin}/${intel_dll}" )
            list( APPEND rad2_runtime_dlls "${rad2_compiler_bin}/${intel_dll}" )
        endif()
    endforeach()

    add_custom_command( TARGET rad2 POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${rad2_runtime_dlls} $<TARGET_FILE_DIR:rad2>
        COMMAND_EXPAND_LISTS
    )
endif()
