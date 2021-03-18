set(TBM_BENCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

function(tbm_add_bench target_name_var bench_name bench_src thrust_target)
  set(bench_target ${bench_name})
  set(${target_name_var} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")
  target_link_libraries(${bench_target} ${thrust_target} nvbench::main fmt::fmt)
  target_include_directories(${bench_target} PRIVATE "${TBM_BENCH_ROOT}")
  set_target_properties(${bench_target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TBM_EXECUTABLE_OUTPUT_DIR}"
  )
endfunction()

function(tbm_add_bench_dir bench_dir)
  foreach(thrust_target IN LISTS THRUST_TARGETS)
    # TODO Need to eventually get this from the props...they might already
    # exist on the targets made with thrust_create_target. If not, they should.
    #  thrust_get_target_property(config_device ${thrust_target} DEVICE)
    #  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

    file(GLOB bench_srcs
      CONFIGURE_DEPENDS
      "${bench_dir}/*.cu"
    )

    # TODO The cmake_path command (3.19) may have better ways to do this:
    file(RELATIVE_PATH bench_prefix "${TBM_BENCH_ROOT}" "${bench_dir}")
    file(TO_CMAKE_PATH "${bench_prefix}" bench_prefix)
    string(REPLACE "/" "." bench_prefix "${bench_prefix}")

    foreach(bench_src IN LISTS bench_srcs)
      get_filename_component(bench_name "${bench_src}" NAME_WLE)
      string(PREPEND bench_name "bench.${bench_prefix}.")
      tbm_add_bench(bench_target ${bench_name} "${bench_src}" ${thrust_target})
    endforeach()
  endforeach()
endfunction()

tbm_get_recursive_subdirs(subdirs)

foreach(subdir IN LISTS subdirs)
  tbm_add_bench_dir("${subdir}")
endforeach()
