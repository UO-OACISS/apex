#---------------------------------------------------------------------------
# Get and build boost

macro(build_boost_project)

include(ExternalProject)

message( "External project - Boost" )

set( Boost_Bootstrap_Command )
if( UNIX )
  set( Boost_Bootstrap_Command ./bootstrap.sh )
  set( Boost_b2_Command ./b2 )
else()
  if( WIN32 )
    set( Boost_Bootstrap_Command bootstrap.bat )
    set( Boost_b2_Command b2.exe )
  endif()
endif()

ExternalProject_Add(project_boost
    #URL "http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.tar.bz2/download"
    #URL "http://sourceforge.net/projects/boost/files/boost/1.59.0/boost_1_59_0.tar.bz2/download"
  URL "http://downloads.sourceforge.net/project/boost/boost/1.59.0/boost_1_59_0.tar.bz2"
  URL_HASH MD5=6aa9a5c6a4ca1016edd0ed1178e3cb87
  BUILD_IN_SOURCE 1
  UPDATE_COMMAND ""
  PATCH_COMMAND ""
  CONFIGURE_COMMAND ${Boost_Bootstrap_Command}
  BUILD_COMMAND ${Boost_b2_Command} install --without-python --disable-icu --prefix=${CMAKE_INSTALL_PREFIX} --threading=multi --link=shared,static --variant=release --without-mpi -j8
  INSTALL_COMMAND ""
  INSTALL_DIR ${CMAKE_INSTALL_PREFIX}
  LOG_DOWNLOAD 1
  LOG_CONFIGURE 1
  LOG_BUILD 1
  LOG_INSTALL 1
)

if( NOT WIN32 )
    set(Boost_LIBRARY_DIR ${CMAKE_INSTALL_PREFIX}/lib/ )
    set(Boost_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/ )
else()
    set(Boost_LIBRARY_DIR ${CMAKE_INSTALL_PREFIX}/lib/ )
    set(Boost_INCLUDE_DIR ${CMAKE_INSTALL_PREFIX}/include/boost-1_59/ )
endif()

set(Boost_LIBRARIES ${Boost_LIBRARY_DIR}libboost_system.so ${Boost_LIBRARY_DIR}libboost_thread.so ${Boost_LIBRARY_DIR}libboost_timer.so ${Boost_LIBRARY_DIR}libboost_regex.so ${Boost_LIBRARY_DIR}libboost_chrono.so)
set(Boost_INCLUDE_DIRS ${Boost_INCLUDE_DIR})
set(Boost_DIR ${CMAKE_INSTALL_PREFIX})

endmacro(build_boost_project)
