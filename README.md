Github-CI:<br>
[![Build Status][github_linux_status]][github_linux_link]
[![Build Status][github_macos_status]][github_macos_link]
[![Build Status][github_windows_status]][github_windows_link]

[![Build Status][github_amd64_docker_status]][github_amd64_docker_link]

[github_linux_status]: ./../../actions/workflows/amd64_linux.yml/badge.svg
[github_linux_link]: ./../../actions/workflows/amd64_linux.yml
[github_macos_status]: ./../../actions/workflows/amd64_macos.yml/badge.svg
[github_macos_link]: ./../../actions/workflows/amd64_macos.yml
[github_windows_status]: ./../../actions/workflows/amd64_windows.yml/badge.svg
[github_windows_link]: ./../../actions/workflows/amd64_windows.yml

[github_amd64_docker_status]: ./../../actions/workflows/amd64_docker.yml/badge.svg
[github_amd64_docker_link]: ./../../actions/workflows/amd64_docker.yml

# Introduction

This is a Modified Google Or-Tools SatRunner 

C++ Project using the [FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)
module.

This project should run on Linux, Mac and Windows.

# CMake Dependencies Tree

This CMake project is composed of one executable SkedaiSatRunner 
with the following dependencies:  
```
ortools: PUBLIC ZLIB abseil-cpp Protobuf re2 Cbc SCIP
SkedaiSatRunner: PRIVATE ortools
```

# C++ Project Build

To build the C++ project, as usual:
```sh
cmake -S. -Bbuild
cmake --build build -v
```

# License

Apache 2. See the LICENSE file for details.

# Disclaimer

This is AiI Tech Knowledge LLC project that was based on 
https://github.com/or-tools/cmake_or-tools.git
