
set(build_shared_libraries TRUE)
find_package (Python3 REQUIRED COMPONENTS Interpreter Development)

set(PUZZLE_PATH_PREFIX ${CMAKE_SOURCE_DIR}/)
set(RLP_PATH_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/../rlp/)

set(platform_common_sources_pygame ${RLP_PATH_PREFIX}pygame.c ${PUZZLE_PATH_PREFIX}printing.c)

foreach(file combi.c divvy.c drawing.c dsf.c findloop.c grid.c latin.c
    laydomino.c loopgen.c malloc.c matching.c midend.c misc.c penrose.c hat.c
    ps.c random.c sort.c tdq.c tree234.c version.c)
    list(APPEND c_sources ${PUZZLE_PATH_PREFIX}${file})
endforeach()


if(build_shared_libraries)
    add_library(common_pygame SHARED ${c_sources}
                ${platform_common_sources_pygame})
    target_link_libraries(common_pygame Python3::Python)

    function(puzzle NAME)
      cmake_parse_arguments(OPT
        "" "DISPLAYNAME;DESCRIPTION;OBJECTIVE;WINDOWS_EXE_NAME" "" ${ARGN})

      if(NOT DEFINED OPT_WINDOWS_EXE_NAME)
        set(OPT_WINDOWS_EXE_NAME ${NAME})
      endif()

      if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        set(EXENAME ${OPT_WINDOWS_EXE_NAME})
      else()
        set(EXENAME ${NAME})
      endif()

      set(exename_${NAME} ${EXENAME} PARENT_SCOPE)
      set(displayname_${NAME} ${OPT_DISPLAYNAME} PARENT_SCOPE)
      set(description_${NAME} ${OPT_DESCRIPTION} PARENT_SCOPE)
      set(objective_${NAME} ${OPT_OBJECTIVE} PARENT_SCOPE)

      set(official TRUE)
      if(NAME STREQUAL nullgame)
        # nullgame is not a playable puzzle; it has to be built (to prove
        # it still can build), but not installed, or included in the main
        # list of puzzles, or compiled into all-in-one binaries, etc. In
        # other words, it's not "officially" part of the puzzle
        # collection.
        set(official FALSE)
      endif()
      if(${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}/unfinished)
        # The same goes for puzzles in the 'unfinished' subdirectory,
        # although we make an exception if configured to on the command
        # line.
        list(FIND PUZZLES_ENABLE_UNFINISHED ${NAME} enable_this_one)
        if(enable_this_one EQUAL -1)
          set(official FALSE)
        endif()
      endif()

      if (official)
        set(puzzle_names ${puzzle_names} ${NAME} PARENT_SCOPE)
        set(puzzle_sources ${puzzle_sources} ${CMAKE_CURRENT_SOURCE_DIR}/${NAME}.c PARENT_SCOPE)
      endif()

      get_platform_puzzle_extra_source_files(extra_files ${NAME})

      if (build_individual_puzzles)
        add_executable(${EXENAME} ${NAME}.c ${extra_files})
        target_link_libraries(${EXENAME}
          common ${platform_gui_libs} ${platform_libs})
        set_property(TARGET ${EXENAME} PROPERTY exename ${EXENAME})
        set_property(TARGET ${EXENAME} PROPERTY displayname ${OPT_DISPLAYNAME})
        set_property(TARGET ${EXENAME} PROPERTY description ${OPT_DESCRIPTION})
        set_property(TARGET ${EXENAME} PROPERTY objective ${OPT_OBJECTIVE})
        set_property(TARGET ${EXENAME} PROPERTY official ${official})
        set_platform_puzzle_target_properties(${NAME} ${EXENAME})
        set_platform_gui_target_properties(${EXENAME})
      endif()
      
      if(official)
        add_library(lib${EXENAME} SHARED ${NAME}.c ${extra_files})
        target_link_libraries(lib${EXENAME} common_pygame ${platform_libs})
        set_property(TARGET lib${EXENAME} PROPERTY POSITION_INDEPENDENT_CODE ON)
        set_property(TARGET lib${EXENAME} PROPERTY exename ${EXENAME})
        set_property(TARGET lib${EXENAME} PROPERTY displayname ${OPT_DISPLAYNAME})
        set_property(TARGET lib${EXENAME} PROPERTY description ${OPT_DESCRIPTION})
        set_property(TARGET lib${EXENAME} PROPERTY objective ${OPT_OBJECTIVE})
        set_property(TARGET lib${EXENAME} PROPERTY official ${official})
        set_platform_puzzle_target_properties(${NAME} lib${EXENAME})
        set_platform_gui_target_properties(lib${EXENAME})
      endif()
    endfunction()
endif()
