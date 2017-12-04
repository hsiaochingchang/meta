/**
 * @file kmeans_model.cpp
 * @author Hsiao-Ching Chang
 */

#include "meta/topics/kmeans_model.h"

#include "meta/classify/multiclass_dataset.h"
#include "meta/index/ranker/okapi_bm25.h"
#include "meta/learn/transform.h"
#include "meta/logging/logger.h"
#include <algorithm>
#include <iostream>
#include <cmath>
#include <utility>

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
      num_terms_(inv_idx_->unique_terms()),
      num_docs_(inv_idx_->num_docs())
{
    documents_.resize(num_docs_);
    for (auto& document : documents_)
        document.resize(num_terms_);
    centroids_.resize(num_topics_);
    for (auto& centroid : centroids_)
        centroid.resize(num_terms_);
    topics_.resize(num_docs_);
}

void kmeans_model::init_documents()
{
    LOG(info) << "Creating TF-IDF vectors\n" << ENDLG;
    classify::multiclass_dataset docs{fwd_idx_};
    index::okapi_bm25 ranker;
    learn::tfidf_transform(docs, *inv_idx_, ranker);

    for (const auto& instance : docs)
    {    
        for (const auto& pair : instance.weights)
        {
            const term_id& t_id = pair.first;
            const double weight = pair.second;
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
        LOG(info) << "Initializing model using kmeans++\n" << ENDLG;

        // Select the first centroid.
        std::uniform_int_distribution<uint64_t> uniform_dist(0, num_docs_ - 1);
        const doc_id& d_id = doc_id{uniform_dist(rd)};
        const topic_id& t_id = topic_id{0};
        centroids_[t_id] = documents_[d_id];

        // Select all the centroids left. Each element of weights is the distance
        // to the nearest existing centroid of a document.
        std::vector<double> weights(num_docs_);
        for (uint64_t centroid_count = 1; centroid_count < num_topics_;
                 ++centroid_count)
        {
            // Find closest centroid among existing ones for each document.
            for (doc_id d_id{0}; d_id < num_docs_; ++d_id)
            {
                const auto& pair = find_nearest_cluster(documents_[d_id],
                                                        centroid_count);
                weights[d_id] = pair.second;
            }
            // Select from the weighted distribution.
            std::discrete_distribution<uint64_t> dist(weights.begin(),
                                                      weights.end());
            const doc_id& d_id = doc_id{dist(rd)};
            const topic_id& t_id = topic_id{centroid_count};
            centroids_[t_id] = documents_[d_id];
        }
    }
    else if (init_method == "randk")
    {
        LOG(info) << "Initializing model using randk\n" << ENDLG;

        for (topic_id t_id{0}; t_id < num_topics_; ++t_id)
        {
            std::uniform_int_distribution<uint64_t> uniform_dist(
                0, num_docs_ - 1);
            const doc_id& d_id = doc_id{uniform_dist(rd)};
            centroids_[t_id] = documents_[d_id];
        }
    }
    else
        throw std::runtime_error{"invalid initalization method"};
}

bool kmeans_model::assign_document(doc_id d_id)
{
    const auto& pair = find_nearest_cluster(documents_[d_id]);
    if (pair.first == topics_[d_id])
        return false;
    else
    {
        topics_[d_id] = pair.first;
        return true;
    }
}

void kmeans_model::update_centroids()
{
    std::multimap<topic_id, doc_id> clusters;
    for (doc_id d_id{0}; d_id < num_docs_; ++d_id)
        clusters.insert(std::make_pair(topics_[d_id], d_id));

    // Compute mean for each cluster
    for (topic_id t_id{0}; t_id < num_topics_; ++t_id)
    {
        const auto& d_ids = clusters.equal_range(t_id);
        std::vector<doc_id> doc_ids;
        for (auto it = d_ids.first; it != d_ids.second; ++it)
            doc_ids.push_back(it->second);
        centroids_[t_id] = compute_mean(doc_ids);
    }
}

std::pair<topic_id, double> kmeans_model::find_nearest_cluster(
    const feature& feature)
{
    return find_nearest_cluster(feature, num_topics_);
}

std::pair<topic_id, double> kmeans_model::find_nearest_cluster(
    const feature& feature, size_t cluster_limit)
{
    std::vector<double> distances(cluster_limit);
    for (uint64_t c_id = 0; c_id < cluster_limit; ++c_id)
        distances[c_id] = compute_distance(
            std::make_pair(feature, centroids_[c_id]));

    const auto min_distance = std::min_element(distances.begin(),
                                               distances.end());
    const uint64_t index = std::distance(distances.begin(), min_distance);
    return std::make_pair(topic_id{index}, *min_distance);
}

feature kmeans_model::compute_mean(const std::vector<doc_id>& doc_ids)
{
    if (doc_ids.empty())
        throw std::runtime_error{"cluster cannot be empty"};
        
    feature mean(num_terms_, 0);

    for (const auto& doc_id : doc_ids)
        for (term_id w_id{0}; w_id < num_terms_; ++w_id)
            mean[w_id] += documents_[doc_id][w_id];
    
    for (term_id w_id{0}; w_id < num_terms_; ++w_id)
        mean[w_id] /= doc_ids.size();

    return mean;
}

double kmeans_model::compute_distance(
    const std::pair<feature, feature>& features)
{
    // Sum-of-squares distance.
    double distance = 0.0;
    for (term_id w_id{0}; w_id < num_terms_; ++w_id)
    {
        const double diff = features.first[w_id] - features.second[w_id];
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
            if (assign_document(d_id))
                ++update_count;
        update_centroids();

        LOG(progress) << "Iteration " << i + 1
            << " update " << update_count << " documents\n" << ENDLG;

        if (update_count == 0)
        {
            LOG(progress) << "No new cluster assignment is made\n" << ENDLG;
            break;
        }
    }

    if (num_output_terms > 0)
        print_topics(num_output_terms);
}

void kmeans_model::save_documents(const std::string& filename) const
{
    std::ofstream file{filename};

    for (doc_id d_id{0}; d_id < num_docs_; ++d_id)
    {
        for (term_id w_id{0}; w_id < num_terms_; ++w_id)
            file << documents_[d_id][w_id] << " ";
        file << std::endl;
    }
}

void kmeans_model::save_centroids(const std::string& filename) const
{
    std::ofstream file{filename};

    for (topic_id t_id{0}; t_id < num_topics_; ++t_id)
    {
        for (term_id w_id{0}; w_id < num_terms_; ++w_id)
            file << centroids_[t_id][w_id] << " ";
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

void kmeans_model::print_topics(size_t num_terms)
{
    std::priority_queue<std::pair<double, term_id>> pq;

    for (topic_id t_id{0}; t_id < num_topics_; ++t_id)
    {
        std::cout << "Topic " << t_id + 1 << std::endl
                  << "---" << std::endl;

        for (term_id w_id{0}; w_id < num_terms_; ++w_id)
            pq.push(std::make_pair(centroids_[t_id][w_id], w_id));

        for (uint64_t i = 0; i < num_terms; ++i)
        {
            if (pq.empty())
                break;

            const auto& pair = pq.top();
            std::cout << inv_idx_->term_text(pair.second)
                      << "\t" << pair.first << std::endl;
            pq.pop();
        }
        std::cout << std::endl;
    }
}

size_t kmeans_model::num_topics() const
{
    return num_topics_;
}

size_t kmeans_model::num_terms() const
{
    return num_terms_;
}

size_t kmeans_model::num_docs() const
{
    return num_docs_;
}
}
}
