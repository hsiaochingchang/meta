/**
 * @file kmeans_model.cpp
 * @author Hsiao-Ching Chang
 */

#include "meta/topics/kmeans_model.h"

namespace meta
{
namespace topics
{

kmeans_model::kmeans_model(std::shared_ptr<index::forward_index> idx,
                           std::size_t num_topics)
    : idx_{std::move(idx)},
      num_topics_{num_topics},
      num_words_(idx_->unique_terms())
{
    /* nothing */
}

void kmeans_model::run(uint64_t num_iters)
{
    /* TODO implementation */
    std::cerr << "Run function in kmeans_model." << std::endl;
}

void kmeans_model::save(const std::string& prefix) const
{
    /* TODO implementation */
}

uint64_t kmeans_model::num_topics() const
{
    return num_topics_;
}
}
}
