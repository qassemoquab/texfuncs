
CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.6)
IF(LUAROCKS_PREFIX)
    MESSAGE(STATUS "Installing Torch through Luarocks")
    STRING(REGEX REPLACE "(.*)lib/luarocks/rocks.*" "\\1" CMAKE_INSTALL_PREFIX  "${LUAROCKS_PREFIX}")
    MESSAGE(STATUS "Prefix inferred from Luarocks: ${CMAKE_INSTALL_PREFIX}")
ENDIF()
FIND_PACKAGE(Torch REQUIRED)

FILE(GLOB luasrc *.lua)

#SET(luasrc ${luasrc} test/test.lua)
ADD_TORCH_PACKAGE(nnbhwdtexfuncs "" "${luasrc}" "nnbhwdtexfuncs")


FIND_PACKAGE(CUDA 4.0 REQUIRED)

SET(src-cuda init.cu)


LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_35")

INCLUDE_DIRECTORIES("${Torch_INSTALL_INCLUDE}/THC")
LINK_DIRECTORIES("${Torch_INSTALL_LIB}")

CUDA_ADD_LIBRARY(nnbhwdtexfuncs MODULE ${src-cuda})
TARGET_LINK_LIBRARIES(nnbhwdtexfuncs luaT THC TH)
IF(APPLE)
  SET_TARGET_PROPERTIES(nnbhwdtexfuncs PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
ENDIF()


### Torch packages supposes libraries prefix is "lib"
SET_TARGET_PROPERTIES(nnbhwdtexfuncs PROPERTIES
  PREFIX "lib"
  IMPORT_PREFIX "lib")
    
INSTALL(TARGETS nnbhwdtexfuncs
  RUNTIME DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}"
  LIBRARY DESTINATION "${Torch_INSTALL_LUA_CPATH_SUBDIR}")

INSTALL(
  FILES
  ${luasrc}
  DESTINATION "${Torch_INSTALL_LUA_PATH_SUBDIR}/nnbhwdtexfuncs")



