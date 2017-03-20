add_executable              (swap_vs_alltoallv swap.cpp)
target_link_libraries       (swap_vs_alltoallv ${libraries})

install(TARGETS swap_vs_alltoallv
        DESTINATION ${CMAKE_INSTALL_PREFIX}/communication/swap_vs_alltoallv/
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)

install(FILES SWAP_TEST
        DESTINATION ${CMAKE_INSTALL_PREFIX}/communication/swap_vs_alltoallv/
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_WRITE GROUP_EXECUTE WORLD_READ WORLD_WRITE WORLD_EXECUTE)
