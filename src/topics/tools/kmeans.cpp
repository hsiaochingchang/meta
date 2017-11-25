#include <iostream>
#include <string>
#include <vector>

#include "meta/topics/kmeans_model.h"

#include "cpptoml.h"

#include "meta/caching/no_evict_cache.h"
#include "meta/index/forward_index.h"
#include "meta/logging/logger.h"

using namespace meta;

bool check_parameter(const std::string& file, const cpptoml::table& group,
                     const std::string& param)
{
    if (!group.contains(param))
    {
        std::cerr << "Missing kmeans configuration parameter " << param << " in "
                  << file << std::endl;
        return false;
    }
    return true;
}

int run_kmeans(const std::string& config_file)
{
    using namespace meta::topics;
    auto config = cpptoml::parse_file(config_file);

    if (!config->contains("kmeans"))
    {
        std::cerr << "Missing kmeans configuration group in " << config_file
                  << std::endl;
        return 1;
    }

    auto kmeans_group = config->get_table("kmeans");

    // TODO add parameters and sort in alphabetical order.
    if (!check_parameter(config_file, *kmeans_group, "max-iters")
        || !check_parameter(config_file, *kmeans_group, "topics"))
        return 1;

    auto iters = *kmeans_group->get_as<uint64_t>("max-iters");
    auto topics = *kmeans_group->get_as<std::size_t>("topics");

    auto f_idx
        = index::make_index<index::forward_index, caching::no_evict_cache>(
            *config);
    std::cout << "Beginning K-means clustering..."
              << std::endl;

    kmeans_model model{f_idx, topics};
    model.run(iters);
    // model.save(save_prefix);
    return 0;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        std::cerr << "Usage:\t" << argv[0] << " configFile" << std::endl;
        return 1;
    }

    logging::set_cerr_logging();
    return run_kmeans(argv[1]);
}
