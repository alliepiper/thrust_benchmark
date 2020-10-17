file(GLOB bench_srcs
  RELATIVE "${CMAKE_CURRENT_LIST_DIR}"
  CONFIGURE_DEPENDS
  *.cu *.cpp
)

function(add_bench target_name_var bench_name bench_src thrust_target)
  set(bench_target bench.${bench_name})
  set(${target_name_var} ${bench_target} PARENT_SCOPE)

  add_executable(${bench_target} "${bench_src}")
  target_link_libraries(${bench_target} ${thrust_target} nvbench)
  set_target_properties(${bench_target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TBM_EXECUTABLE_OUTPUT_DIR}"
  )
endfunction()

foreach(thrust_target IN LISTS THRUST_TARGETS)
  # TODO Need to eventually get this from the props...they might already
  # exist on the targets made with thrust_create_target. If not, they should.
  #  thrust_get_target_property(config_device ${thrust_target} DEVICE)
  #  thrust_get_target_property(config_prefix ${thrust_target} PREFIX)

  foreach(bench_src IN LISTS bench_srcs)
    get_filename_component(bench_name "${bench_src}" NAME_WLE)
    add_bench(bench_target ${bench_name} "${bench_src}" ${thrust_target})
  endforeach()
endforeach()
