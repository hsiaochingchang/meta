project(meta-topics)

add_subdirectory(tools)

add_library(meta-topics lda_cvb.cpp
                        lda_gibbs.cpp
                        lda_model.cpp
                        lda_scvb.cpp
                        parallel_lda_gibbs.cpp
                        kmeans_model.cpp)
target_link_libraries(meta-topics meta-index meta-classify meta-ranker)

install(TARGETS meta-topics
        EXPORT meta-exports
        DESTINATION lib)
