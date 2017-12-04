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

#include <vector>

#include "meta/config.h"
#include "meta/index/forward_index.h"
#include "meta/index/inverted_index.h"

MAKE_NUMERIC_IDENTIFIER(topic_id, uint64_t)

namespace meta
{
namespace topics
{

using feature = std::vector<double>;

/**
 * A K-Means topic model base class.
 *
 * Required config parameters (for use with the ./kmeans executable):
 * ~~~toml
 * max-iters = 1000
 * topics = 2
 * init-method = "kmeans++" # randk
 * output-terms = 8
 * model-prefix = "kmeans-model"
 * ~~~
 *
 * Optional config parameters: none.
 */
class kmeans_model
{
  public:
    /**
     * Constructs a kmeans_model over the given indices of documents and
     * with a fixed number of topics.
     *
     * @param fwd_idx The forward index of the corpus
     * @param inv_idx The inverted index of the corpus
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
     * Runs the model for a given number of iterations, or until no update
     * is made in an iteration. Currently, two initialization methods are
     * implemented, random-k-points or kmeans++. The method could be chosen
     * by specifying either "randk" or "kmeans++".
     *
     * @param num_iters The maximum allowed number of iterations
     * @param init_method The preferred initialization method to use
     * @param num_output_terms The number of output terms to be shown after
     * model fitting
     */
    void run(uint64_t num_iters, std::string init_method,
             uint64_t num_output_terms);

    /**
     * Saves the current model to a set of files beginning with prefix:
     * prefix.docs, prefix.centroids, and prefix.clusters.
     *
     * @param prefix The prefix for all generated files over this model
     */
    void save(const std::string& prefix) const;

    /**
     * Prints the clustering results. Each topic is shown along with the
     * most significant terms within it.
     *
     * @param num_terms The number of terms that are shown with the topics
     */
    void print_topics(size_t num_terms);

    /**
     * @return the number of topics in this model
     */
    size_t num_topics() const;

    /**
     * @return the number of terms of the corpus
     */
    size_t num_terms() const;

    /**
     * @return the number of documents of the corpus
     */
    size_t num_docs() const;

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
     * Extracts the document vectors. Performs TF-IDF transformation using
     * the inverted index and stores the vectors to the model.
     */
    void init_documents();

    /**
     * Randomly initializes the centroids of the clusters.
     *
     * @param init_method The name of initialization method to be used
     */
    void init_centroids(const std::string& init_method);

    /**
     * Assigns a document to its nearest cluster.
     *
     * @param d_id The document to assign cluster to
     * @return true if the cluster assignment of the document has been
     * changed
     */
    bool assign_document(doc_id d_id);

    /**
     * Computes the new centroids by calculating the new means of the
     * clusters.
     */
    void update_centroids();

    /**
     * Find the nearest cluster for a given document vector. Searches
     * through all num_topics_ clusters.
     *
     * @param feature The document vector to find cluster for
     * @return the topic_id and the distance of the nearest centroid
     */
    std::pair<topic_id, double> find_nearest_cluster(
        const feature& feature);

    /**
     * Finds the nearest cluster for a given document vector. Searches
     * through only the first m clusters. The function is useful during
     * kmeans++ initialization.
     *
     * @param feature The document vector to find cluster for
     * @param cluster_limit The number of clusters to search within
     * @return the topic_id and the distance of the nearest centroid
     */
    std::pair<topic_id, double> find_nearest_cluster(
        const feature& feature, size_t cluster_limit);

    /**
     * Computes the mean vector over a set of documents.
     *
     * @param doc_ids The set of documents, represented in doc_id
     * @return the mean vector
     */
    feature compute_mean(const std::vector<doc_id>& doc_ids);

    /**
     * The sum-of-squares distance function between feature vectors.
     *
     * @param features The pair of feature vectors
     * @return the sum-of-squares distance
     */
    double compute_distance(const std::pair<feature, feature>& features);

    /**
     * Saves the document vectors to disk.
     *
     * @param filename The filename to save to
     */
    void save_documents(const std::string& filename) const;

    /**
     * Saves the centroid vectors to disk.
     *
     * @param filename The filename to save to
     */
    void save_centroids(const std::string& filename) const;

    /**
     * Saves the topic assignments to disk.
     *
     * @param filename The filename to save to
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
     * The feature vectors of all documents in a dense matrix.
     */
    std::vector<feature> documents_;

    /**
     * The centroids of each cluster.
     */
    std::vector<feature> centroids_;

    /**
     * The assigned topic ids for each document.
     */
    std::vector<topic_id> topics_;

    /**
     * The number of topics.
     */
    std::size_t num_topics_;

    /**
     * The number of total unique words.
     */
    std::size_t num_terms_;

    /**
     * The number of documents.
     */
    std::size_t num_docs_;
};
}
}

#endif
