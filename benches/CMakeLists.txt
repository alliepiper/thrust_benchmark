set(TBM_CTEST_DEVICE "all" CACHE STRING
  "--device <arg> to use when running benchmarks with ctest"
)
set(TBM_CTEST_TIMEOUT "15.0" CACHE STRING
  "--timeout <arg> to use when running benchmarks with ctest (seconds)"
)
set(TBM_CTEST_MAX_NOISE "0.5" CACHE STRING
  "--max-noise <arg> to use when running benchmarks with ctest"
)
option(TBM_CTEST_OUTPUT_MARKDOWN
  "Write markdown logs when running benchmarks with ctest." ON
)
option(TBM_CTEST_OUTPUT_CSV
  "Write csv output when running benchmarks with ctest." ON
)
option(TBM_CTEST_OUTPUT_JSON
  "Write json files when running benchmarks with ctest." ON
)

set(TBM_BENCH_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# Metatarget to build all benchmarks
add_custom_target(bench.all)

function(tbm_add_bench target_name_var bench_name bench_src thrust_target)
  set(bench_target ${bench_name})
  set(${target_name_var} ${bench_target} PARENT_SCOPE)

  # To support compilation with `nvcc -keep`, all source files must have unique
  # filenames, not just unique paths, as nvcc's temporary files are all prefixed
  # with the source filename and written to the same directory.
  #
  # To work-around this, configure a wrapper file with the full test name and
  # pass that to the compiler.
  set(wrapped_src "${CMAKE_CURRENT_BINARY_DIR}/wrapped_srcs/${bench_name}.cu")
  file(CONFIGURE
    OUTPUT "${wrapped_src}"
    CONTENT "#include <@bench_src@>\n"
    @ONLY
  )

  set(results_dir "${TBM_BINARY_DIR}/results")
  file(MAKE_DIRECTORY "${results_dir}")

  add_executable(${bench_target} "${wrapped_src}")
  target_link_libraries(${bench_target} ${thrust_target} nvbench::main fmt::fmt)
  target_include_directories(${bench_target} PRIVATE "${TBM_BENCH_ROOT}")
  set_target_properties(${bench_target} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${TBM_LIBRARY_OUTPUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${TBM_EXECUTABLE_OUTPUT_DIR}"
  )
  add_dependencies(bench.all ${bench_target})

  set(test_args
    "--device" "${TBM_CTEST_DEVICE}"
    "--timeout" "${TBM_CTEST_TIMEOUT}"
    "--max-noise" "${TBM_CTEST_MAX_NOISE}"
  )
  if (TBM_CTEST_OUTPUT_MARKDOWN)
    list(APPEND test_args --md "${results_dir}/${bench_name}.log.md")
  endif()
  if (TBM_CTEST_OUTPUT_CSV)
    list(APPEND test_args --csv "${results_dir}/${bench_name}.csv")
  endif()
  if (TBM_CTEST_OUTPUT_JSON)
    list(APPEND test_args --json "${results_dir}/${bench_name}.json")
  endif()
  add_test(NAME ${bench_name}
    COMMAND "$<TARGET_FILE:${bench_target}>" "${test_args}"
    COMMAND_EXPAND_LISTS
  )

  # Build a list of labels with each level of the benchmark path. CTest will use
  # these to print a summary of time spent benchmarking each
  # project/algorithm/etc.
  #
  # e.g. for benchmark: bench.cub.device.histogram.even
  # -> bench;bench.cub;bench.cub.device;bench.cub.device.histogram
  set(test_labels)
  # Split benchmark name into list, e.g. bench;cub;device;histogram
  string(REPLACE "." ";." label_flotsam "${bench_name}")
  list(POP_BACK label_flotsam)
  set(test_label "")
  foreach (label_component IN LISTS label_flotsam)
    string(APPEND test_label ${label_component})
    list(APPEND test_labels ${test_label})
  endforeach()
  set_tests_properties(${bench_name} PROPERTIES
    RUN_SERIAL ON
    LABELS "${test_labels}"
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
