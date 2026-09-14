if(NOT WIN32)
  message(FATAL_ERROR
      "NeuronBridge currently supports Windows x64 only. "
      "Linux support is planned for a later release.")
endif()

if(NOT MSVC)
  message(FATAL_ERROR
      "The initial NeuronBridge release requires Visual Studio 2022/MSVC.")
endif()

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "NeuronBridge requires a 64-bit build.")
endif()
