/**
 * @file kmeans_model.cpp
 * @author Hsiao-Ching Chang
 */

#include "meta/topics/kmeans_model.h"

#include "meta/classify/multiclass_dataset.h"
#include "meta/index/ranker/okapi_bm25.h"
#include "meta/learn/transform.h"

#include <cmath>
#include <utility>
#include <algorithm>

namespace meta
{
namespace topics
{

kmeans_model::kmeans_model(std::shared_ptr<index::forward_index> fwd_idx,
                           std::shared_ptr<index::inverted_index> inv_idx,
                           std::size_t num_topics)
    : fwd_idx_{std::move(fwd_idx)},
      inv_idx_{std::move(inv_idx)},
      num_topics_(num_topics),
      num_words_(inv_idx_->unique_terms()),
      num_docs_(inv_idx_->num_docs())
{
    documents_.resize(num_docs_);
    for (auto& document : documents_)
        document.resize(num_words_);
    centroids_.resize(num_topics_);
    for (auto& centroid : centroids_)
        centroid.resize(num_words_);
    topics_.resize(num_docs_);
}

void kmeans_model::init_documents()
{
    std::cout << "Creating tf-idf vectors" << std::endl;
    classify::multiclass_dataset docs{fwd_idx_};
    index::okapi_bm25 ranker;
    learn::tfidf_transform(docs, *inv_idx_, ranker);

    std::cout << "Storing tf-idf vectors" << std::endl;
    for (const auto& instance : docs)
    {    
        for (const auto& pair : instance.weights)
        {
            term_id t_id = pair.first;
            double weight = pair.second;
            documents_[instance.id][t_id] = weight;
        }
    }
}

void kmeans_model::init_centroids(const std::string& init_method)
{
    std::random_device rd;
    std::mt19937 random_engine(rd());

    if (init_method == "kmeans++")
    {
        // Select the first centroid.
        std::uniform_int_distribution<uint64_t> uniform_dist(0, num_docs_ - 1);
        uint64_t d_id = uniform_dist(rd);
        centroids_[0] = documents_[d_id];

        // Select all the centroids left. Each element of weights is the distance
        // to the nearest existing centroid of a document.
        std::vector<double> weights(num_docs_);
        for (uint64_t centroid_count = 1; centroid_count < num_topics_;
                 ++centroid_count)
        {
            // Find closest centroid among existing ones for each document.
            for (uint64_t d_id = 0; d_id < num_docs_; ++d_id)
            {
                const auto& pair = find_nearest_cluster(documents_[d_id],
                                                        centroid_count);
                weights[d_id] = pair.second;
            }
            // Select from the weighted distribution.
            std::discrete_distribution<uint64_t> dist(
                weights.begin(), weights.end());
            uint64_t d_id = dist(rd);
            centroids_[centroid_count] = documents_[d_id];
        }
        std::cout << "Initialized " << centroids_.size() << " centroids" << std::endl;
    }
    else
    {
        for (uint64_t i = 0; i < num_topics_; ++i)
        {
            // Select the first centroid.
            std::uniform_int_distribution<uint64_t> uniform_dist(0, num_docs_ - 1);
            uint64_t d_id = uniform_dist(rd);
            centroids_[i] = documents_[d_id];
        }
    }
}

bool kmeans_model::assign_document(doc_id d_id)
{
    const auto& pair = find_nearest_cluster(documents_[d_id]);
    if (pair.first == topics_[d_id])
    {
        return false;
    }
    else
    {
        topics_[d_id] = pair.first;
        return true;
    }
}

void kmeans_model::update_centroids()
{
    std::multimap<uint64_t, doc_id> clusters;
    for (doc_id d_id{0}; d_id < num_docs_; ++d_id)
        clusters.insert(std::make_pair(topics_[d_id], d_id));

    // Compute mean for each cluster
    for (uint64_t c_id = 0; c_id < num_topics_; ++c_id)
    {
        const auto& d_ids = clusters.equal_range(c_id);
        std::vector<uint64_t> doc_ids;
        for (auto it = d_ids.first; it != d_ids.second; ++it)
            doc_ids.push_back(it->second);
        std::cout << "Cluster " << c_id + 1 << " contains "
                  << doc_ids.size() << " docs" << std::endl;
        centroids_[c_id] = compute_mean(doc_ids);
    }
    std::cout << std::endl;
}

std::pair<uint64_t, double> kmeans_model::find_nearest_cluster(
    const Feature& feature)
{
    return find_nearest_cluster(feature, num_topics_);
}

std::pair<uint64_t, double> kmeans_model::find_nearest_cluster(
    const Feature& feature, uint64_t centroid_count)
{
    std::vector<double> distances(centroid_count);
    for (uint64_t c_id = 0; c_id < centroid_count; ++c_id)
    {
        distances[c_id] = compute_distance(
            std::make_pair(feature, centroids_[c_id]));
    }
    auto min_distance = std::min_element(distances.begin(), distances.end());
    return std::make_pair(std::distance(distances.begin(), min_distance),
                          *min_distance);
}

Feature kmeans_model::compute_mean(const std::vector<uint64_t>& doc_ids)
{
    // TODO sanity check doc_ids non-empty

    Feature mean(num_words_, 0);

    for (const auto& doc_id : doc_ids)
        for (uint64_t idx = 0; idx < num_words_; ++idx)
            mean[idx] += documents_[doc_id][idx];
    
    for (uint64_t idx = 0; idx < num_words_; ++idx)
        mean[idx] /= doc_ids.size();

    return mean;
}

double kmeans_model::compute_distance(const std::pair<Feature, Feature>& features)
{
    double distance = 0.0;
    // Euclidean distance
    for (uint64_t idx = 0; idx < num_words_; ++idx)
    {
        const double diff = features.first[idx] - features.second[idx];
        distance += diff * diff;
    }
    return distance;
}

void kmeans_model::run(uint64_t num_iters, std::string init_method,
                       uint64_t num_output_terms)
{
    init_documents();
    init_centroids(init_method);

    for (uint64_t i = 0; i < num_iters; ++i)
    {
        uint64_t update_count = 0;
        for (doc_id d_id{0}; d_id < num_docs_; ++d_id)
        {
            if (assign_document(d_id))
                ++update_count;
        }
        update_centroids();
        std::cout << "Iteration " << i + 1 << ", update " << update_count
                  << " docs" << std::endl;
        if (update_count == 0)
            break;
    }

    if (num_output_terms > 0)
        print_topics(num_output_terms);
}

void kmeans_model::save_documents(const std::string& filename) const
{
    std::ofstream file{filename};

    for (doc_id d_id{0}; d_id < num_docs_; ++d_id)
    {
        for (term_id t_id{0}; t_id < num_words_; ++t_id)
            file << documents_[d_id][t_id] << " ";
        file << std::endl;
    }
}

void kmeans_model::save_centroids(const std::string& filename) const
{
    std::ofstream file{filename};

    for (uint64_t c_id = 0; c_id < num_topics_; ++c_id)
    {
        for (term_id t_id{0}; t_id < num_words_; ++t_id)
            file << centroids_[c_id][t_id] << " ";
        file << std::endl;
    }
}

void kmeans_model::save_clusters(const std::string& filename) const
{
    std::ofstream file{filename};

    for (doc_id d_id{0}; d_id < num_docs_; ++d_id)
    {
        file << d_id << " " << topics_[d_id] << std::endl;
    }
}

void kmeans_model::save(const std::string& prefix) const
{
    save_documents(prefix + ".docs");
    save_centroids(prefix + ".centroids");
    save_clusters(prefix + ".clusters");
}

void kmeans_model::print_topics(uint64_t num_terms)
{
    std::priority_queue<std::pair<double, term_id>> pq;

    // Print most significant words in each cluster
    for (uint64_t c_id = 0; c_id < num_topics_; ++c_id)
    {
        std::cout << "Cluster " << c_id + 1 << std::endl;
        for (term_id t_id{0}; t_id < num_words_; ++t_id)
            pq.push(std::make_pair(centroids_[c_id][t_id], t_id));

        for (uint64_t i = 0; i < num_terms; ++i)
        {
            if (pq.empty())
                break;
            const auto& pair = pq.top();
            std::cout << inv_idx_->term_text(pair.second) << "\t" << pair.first << std::endl;
            pq.pop();
        }
        std::cout << std::endl;
    }

}

uint64_t kmeans_model::num_topics() const
{
    return num_topics_;
}
}
}
