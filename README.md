<div align="center">
  <h3><a href="https://github.com/kusabana">
    ~kusabana/</a>rad2
  </h3>

Intel® Embree based Source Engine 2013 Lightmap Compiler
</div>

> [!NOTE]  
> SYCL Backend is available for **Intel Arc / Xe GPUs only**

A lot of comments have references to the VRAD source code which can be found at [source-sdk-2013](https://github.com/ValveSoftware/source-sdk-2013)

Not implemented yet: texture lights, _minlight, bumped lighting, leaf ambient cubes, lightstyles, static prop lighting

## Requirements

- [Intel oneAPI DPC++/C++ Compiler **2025.2**](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html) (with oneTBB)
- (Linux) [intel-compute-runtime](https://github.com/intel/compute-runtime)

## Building

### Windows

```
call "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Linux

```
source /opt/intel/oneapi/setvars.sh
cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=icpx -DCMAKE_BUILD_TYPE=Release
cmake --build build
```
