cmake_minimum_required(VERSION 3.22.1)
project(etar_android LANGUAGES CXX)

add_library(etar_core STATIC etar_core.cpp)
target_compile_features(etar_core PUBLIC cxx_std_17)
target_compile_options(etar_core PRIVATE -O3 -ffast-math -fno-math-errno)

add_library(etar_jni SHARED native-lib.cpp)
target_compile_features(etar_jni PRIVATE cxx_std_17)
target_link_libraries(etar_jni PRIVATE etar_core android log)
