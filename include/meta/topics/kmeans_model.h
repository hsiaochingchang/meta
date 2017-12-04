/**
 * @file topics/kmeans_model.h
 * @author Hsiao-Ching Chang
 *
 * All files in META are dual-licensed under the MIT and NCSA licenses. For more
 * details, consult the file LICENSE.mit and LICENSE.ncsa in the root of the
 * project.
 */

#ifndef META_TOPICS_KMEANS_MODEL_H_
#define META_TOPICS_KMEANS_MODEL_H_

// TODO: remove this.
#include <iostream>
#include <vector>

#include "meta/config.h"
#include "meta/index/forward_index.h"
#include "meta/index/inverted_index.h"

MAKE_NUMERIC_IDENTIFIER(topic_id, uint64_t)

namespace meta
{
namespace topics
{

typedef std::vector<double> Feature;

/**
 * TODO: Fix documentation.
 * An LDA topic model base class.
 *
 * Required config parameters (for use with the ./lda executable):
 * ~~~toml
 * inference = "inference-method" # gibbs, pargibbs, cvb, scvb
 * max-iters = 1000
 * alpha = 1.0
 * beta = 1.0
 * topics = 4
 * model-prefix = "prefix"
 * ~~~
 *
 * Optional config parameters: none.
 */
class kmeans_model
{
  public:
    /**
     * Constructs an lda_model over the given set of documents and with a
     * fixed number of topics.
     *
     * @param idx The index containing the documents to use for the model
     * @param num_topics The number of topics to find
     */
    kmeans_model(std::shared_ptr<index::forward_index> fwd_idx,
                 std::shared_ptr<index::inverted_index> inv_idx,
                 std::size_t num_topics);

    /**
     * Destructor. Made virtual to allow for deletion through pointer to
     * base.
     */
    virtual ~kmeans_model() = default;

    /**
     * Runs the model for a given number of iterations, or until a
     * convergence criteria is met.
     *
     * @param num_iters The maximum allowed number of iterations
     * @param convergence The convergence criteria (this has different
     * meanings for different subclass models)
     */
    void run(uint64_t num_iters, std::string init_method,
             uint64_t num_output_terms);

    /**
     * Saves the current model to a set of files beginning with prefix:
     * prefix.phi, prefix.theta, and prefix.terms.
     *
     * @param prefix The prefix for all generated files over this model
     */
    void save(const std::string& filename) const;

    /**
     *
     */
    void print_topics(uint64_t num_terms);

    /**
     * @return the number of topics in this model
     */
    uint64_t num_topics() const;

  protected:
    /**
     * kmeans_models cannot be copy assigned.
     */
    kmeans_model& operator=(const kmeans_model&) = delete;

    /**
     * kmeans_models cannot be copy constructed.
     */
    kmeans_model(const kmeans_model&) = delete;

    /**
     * Extract the document vectors.
     */
    void init_documents();

    /**
     * Randomly generate the centroid of each cluster.
     */
    void init_centroids(const std::string& init_method);

    /**
     * Assign the document to its nearest cluster.
     *
     * @param d_id The document to assign.
     * @return true if the cluster of the document has been changed.
     */
    bool assign_document(doc_id d_id);

    /**
     * Compute the new centroids.
     */
    void update_centroids();

    /**
     * Find the nearest cluster for a given feature vector.
     */
    std::pair<uint64_t, double> find_nearest_cluster(
        const Feature& feature);

    /**
     * Find the nearest cluster for a given feature vector.
     */
    std::pair<uint64_t, double> find_nearest_cluster(
        const Feature& feature, uint64_t centroid_count);
    /**
     *
     */
    Feature compute_mean(const std::vector<uint64_t>& doc_ids);

    /**
     * The distance function between feature vectors.
     */
    double compute_distance(const std::pair<Feature, Feature>& features);

    /**
     *
     */
    void save_documents(const std::string& filename) const;

    /**
     *
     */
    void save_centroids(const std::string& filename) const;

    /**
     *
     */
    void save_clusters(const std::string& filename) const;

    /**
     * The index containing the documents for the model.
     */
    std::shared_ptr<index::inverted_index> inv_idx_;

    /**
     * The index containing the documents for the model.
     */
    std::shared_ptr<index::forward_index> fwd_idx_;

    /**
     *
     */
    std::vector<Feature> documents_;

    /**
     * The centroids of each cluster.
     */
    std::vector<Feature> centroids_;

    /**
     * The assigned topic ids for each document.
     */
    std::vector<uint64_t> topics_;

    /**
     * The number of topics.
     */
    std::size_t num_topics_;

    /**
     * The number of total unique words.
     */
    std::size_t num_words_;

    /**
     * The number of documents.
     */
    std::size_t num_docs_;
};
}
}

#endif
