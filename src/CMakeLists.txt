set_source_files_properties(${CMAKE_CURRENT_SOURCE_DIR}/trace.cu
        PROPERTIES CUDA_SOURCE_PROPERTY_FORMAT OBJ)

OPTIX_add_sample_executable(cuda-sdf target_name
        main.cpp trace.cu trace.h camera.h cuda_check.h vec_math_helper.h)

target_link_libraries(${target_name} ${CUDA_LIBRARIES} sutil_7_sdk stb)