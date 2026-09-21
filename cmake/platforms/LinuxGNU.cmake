if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
  message(FATAL_ERROR "LinuxGNU.cmake requires Linux")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "NeuronBridge requires a 64-bit build")
endif()

if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|amd64|AMD64)$")
  message(FATAL_ERROR
      "The initial Linux release requires x86-64; found ${CMAKE_SYSTEM_PROCESSOR}")
endif()

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  message(FATAL_ERROR
      "Linux NeuronBridge builds require GCC or Clang; found "
      "${CMAKE_CXX_COMPILER_ID}")
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND
   CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11)
  message(FATAL_ERROR "NeuronBridge requires GCC 11 or newer on Linux")
endif()

find_package(Threads REQUIRED)

message(STATUS
    "NeuronBridge Linux compiler: ${CMAKE_CXX_COMPILER_ID} "
    "${CMAKE_CXX_COMPILER_VERSION}")
