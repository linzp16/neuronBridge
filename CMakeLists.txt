cmake_minimum_required(VERSION 3.24)

project(NeuronBridge VERSION 0.1.0 LANGUAGES CXX)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(NeuronBridgeOptions)
include(platforms/WindowsMSVC)

message(STATUS "NeuronBridge repository bootstrap stage")
message(STATUS "Full version: ${NR_VERSION_FULL}")
message(STATUS "Core source migration: pending")
message(STATUS "Bundled dependency migration: pending")
