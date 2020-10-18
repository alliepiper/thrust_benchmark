function(tbm_get_recursive_subdirs subdirs_var)
  set(test_dirs)
  file(GLOB_RECURSE contents
    CONFIGURE_DEPENDS
    LIST_DIRECTORIES ON
    "${CMAKE_CURRENT_LIST_DIR}/*"
  )

  foreach(test_dir IN LISTS contents)
    if(IS_DIRECTORY "${test_dir}")
      list(APPEND test_dirs "${test_dir}")
    endif()
  endforeach()

  set(${subdirs_var} "${test_dirs}" PARENT_SCOPE)
endfunction()
