// EGBI - Basic Usage Example
// Demonstrates building an index, fuzzy search, and serialization.

#include <egbi/egbi.hpp>
#include <iostream>
#include <string>
#include <vector>

int main() {
    // ---- 1. Prepare a corpus ----
    std::vector<std::string> corpus = {
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox leaps over a sleepy dog",
        "A fast dark fox vaults across a tired hound",
        "Machine learning and deep neural networks",
        "Natural language processing with transformers",
        "Convolutional neural networks for image recognition",
        "The lazy dog sleeps in the warm sunshine",
        "Quick brown foxes are faster than lazy dogs",
        "Deep learning models require large datasets",
        "Transformers have revolutionized NLP tasks",
    };

    // ---- 2. Configure and build index ----
    egbi::Config cfg;
    cfg.bit_length = 1024;   // 1024-bit vectors (128 bytes each)
    cfg.alpha      = 2.0f;   // dynamic bit allocation scaling
    cfg.ngram_size = 3;      // character trigrams

    egbi::EGBIIndex index(cfg);
    index.add_documents(corpus);       // builds IDF + indexes all at once

    std::cout << "Indexed " << index.size() << " documents.\n\n";

    // ---- 3. Fuzzy search ----
    std::string query = "quick brown fox jumps";
    float threshold = 0.15f;

    std::cout << "Query: \"" << query << "\"  (threshold >= "
              << threshold << ")\n";
    std::cout << std::string(60, '-') << "\n";

    auto results = index.search(query, threshold, 5);
    for (const auto& r : results) {
        std::cout << "  DocId=" << r.doc_id
                  << "  score=" << r.score
                  << "  | " << corpus[r.doc_id] << "\n";
    }

    // ---- 4. Pairwise similarity ----
    std::cout << "\nPairwise similarities:\n";
    std::cout << "  doc[0] vs doc[1] = " << index.similarity(0, 1) << "\n";
    std::cout << "  doc[0] vs doc[3] = " << index.similarity(0, 3) << "\n";
    std::cout << "  doc[3] vs doc[5] = " << index.similarity(3, 5) << "\n";

    // ---- 5. Save / Load ----
    const std::string path = "demo_index.egbi";
    if (index.save(path)) {
        std::cout << "\nIndex saved to " << path << "\n";

        egbi::EGBIIndex loaded;
        if (loaded.load(path)) {
            std::cout << "Loaded index: " << loaded.size() << " documents.\n";
            // Verify a search still works on the loaded index
            auto r2 = loaded.search(query, threshold, 3);
            std::cout << "Top result after reload: DocId="
                      << r2[0].doc_id << " score=" << r2[0].score << "\n";
        }
    }

    return 0;
}
