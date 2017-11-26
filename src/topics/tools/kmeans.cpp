#include <iostream>
#include <string>
#include <vector>

#include "meta/topics/kmeans_model.h"

#include "cpptoml.h"

#include "meta/analyzers/analyzer.h"
#include "meta/analyzers/tokenizers/icu_tokenizer.h"
#include "meta/analyzers/filters/all.h"
#include "meta/caching/no_evict_cache.h"
#include "meta/corpus/corpus_factory.h"
#include "meta/index/forward_index.h"
#include "meta/io/filesystem.h"
#include "meta/logging/logger.h"

using namespace meta;

/**
 * TODO add documentation.
 */
std::unique_ptr<meta::analyzers::token_stream> create_preprocess_stream(
    const cpptoml::table& config)
{
    using namespace meta::analyzers;
    auto stopwords = config.get_as<std::string>("stop-words");

    std::unique_ptr<token_stream> stream
        = make_unique<tokenizers::icu_tokenizer>();
    stream = make_unique<filters::lowercase_filter>(std::move(stream));
    stream = make_unique<filters::list_filter>(std::move(stream), *stopwords);
    stream = make_unique<filters::porter2_filter>(std::move(stream));
    stream = make_unique<filters::empty_sentence_filter>(std::move(stream));

    return stream;
}

// Read file content.
std::string get_content(const std::string& content_path)
{
    std::ifstream input{content_path};
    std::ostringstream oss;
    oss << input.rdbuf();
    std::string content{oss.str()};
    std::replace_if(content.begin(), content.end(), [](char ch) {
        return ch == '\n' || ch == '\t'; }, ' ');
    return content;
}

template <class Stream>
void preprocess_content(std::string& content, Stream& stream,
                        std::ofstream& outfile)
{
    stream->set_content(std::move(content));
    while (*stream)
    {
        auto next = stream->next();
        if (next == "<s>" || next == "</s>" || next == " ")
            continue;
        else
            outfile << next << " ";
    }
}

// Create line corpus, labels, and names.
void create_line_corpus(const std::string& filename,
                        const std::string& new_filename,
                        const std::string& prefix,
                        const std::string& dataset,
                        std::unique_ptr<meta::analyzers::token_stream> stream)
{
    std::ifstream input_paths{filename};
    if (!input_paths.good())
        std::cerr << "Failed to open " << filename << std::endl;
    std::ofstream content{new_filename};
    std::ofstream labels{new_filename + ".labels"};
    std::ofstream names{new_filename + ".names"};

    uint64_t num_lines = filesystem::num_lines(filename);
    uint64_t cur_line = 0;
    std::cout << "Found " << num_lines << " files" << std::endl;

    std::string path;
    std::string label;
    while (input_paths >> label >> path)
    {
        std::string text = get_content(prefix + "/" + dataset + "/" + path);
        preprocess_content(text, stream, content);
        content << "\n";
        labels << label << "\n";
        names << path << "\n";
        std::cout << ++cur_line << "/" << num_lines << " " << path
                  << "\t\t\t\t\r";
    }
    std::cout << std::endl;
}

// Generate stemmed/stopword-removed line corpus.
std::unique_ptr<meta::corpus::corpus> generate_corpus(
    const std::string& prefix, const std::string& dataset,
    const cpptoml::table& config)
{
    auto stream = create_preprocess_stream(config);
    std::string file =
        prefix + "/" + dataset + "/" + dataset + "-full-corpus.txt";
    // Line corpus.
    std::string new_file = prefix + "/" + dataset + "/" + dataset + ".dat";

    create_line_corpus(file, new_file, prefix, dataset, std::move(stream));

    // Create line_corpus object.
    using namespace meta::corpus;
    auto corp = make_corpus(config);
    std::cout << "Created line corpus with " << corp->size() << " files"
              << std::endl;
    return corp;
}

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

int run_kmeans(std::shared_ptr<cpptoml::table> config,
               std::unique_ptr<meta::corpus::corpus> corp)
{
    using namespace meta::topics;

    if (!config->contains("kmeans"))
    {
        std::cerr << "Missing kmeans configuration group in config"
                  << std::endl;
        return 1;
    }

    auto kmeans_group = config->get_table("kmeans");

    // TODO add parameters and sort in alphabetical order.
    // if (!check_parameter(config_file, *kmeans_group, "max-iters")
    //     || !check_parameter(config_file, *kmeans_group, "topics"))
    //     return 1;

    auto iters = *kmeans_group->get_as<uint64_t>("max-iters");
    auto topics = *kmeans_group->get_as<std::size_t>("topics");
    auto index = *config->get_as<std::string>("index");

    auto f_idx
        = index::make_index<index::forward_index, caching::no_evict_cache>(
            *config, *corp);
    std::cout << "Created forward index for " << corp->size() << " docs"
              << std::endl;
    std::cout << "Index name: " << f_idx->index_name() << std::endl
              << "Unique terms: " << f_idx->unique_terms() << std::endl
              << "Num of docs: " << f_idx->num_docs() << std::endl;
 
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
        std::cerr << "Usage:\t" << argv[0] << " configFile"
                  << std::endl;
        return 1;
    }

    auto config = cpptoml::parse_file(argv[1]);
    auto prefix = *config->get_as<std::string>("prefix");
    auto dataset = *config->get_as<std::string>("dataset");

    logging::set_cerr_logging();
    // Generate line corpus.
    auto corp = generate_corpus(prefix, dataset, *config);
    return run_kmeans(config, std::move(corp));

    return 0;
}
