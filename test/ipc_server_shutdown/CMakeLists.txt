add_renodx_runtime_executable(ipc_server_shutdown_tests tests.cpp)
add_dependencies(renodx_test_suites ipc_server_shutdown_tests)

add_test(NAME ipc_server_shutdown COMMAND ipc_server_shutdown_tests)
set_tests_properties(ipc_server_shutdown PROPERTIES
  LABELS "ipc;shutdown;unit"
  TIMEOUT 10
)
