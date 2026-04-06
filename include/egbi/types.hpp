#pragma once
// EGBI - Entropy-Gated Bit-Vector Indexing
// Types, configuration, and constants.

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>

namespace egbi {

// ---------- Configuration ----------

struct Config {
    uint32_t bit_length = 1024;   // k: total bits in each document vector
    float    alpha      = 2.0f;   // scaling constant for dynamic bit allocation
    uint32_t ngram_size = 3;      // character-level n-gram window size
    uint32_t min_bits   = 1;      // floor on bits per feature
    uint32_t max_bits   = 32;     // ceiling on bits per feature
    float    idf_floor  = 0.0f;   // features with IDF below this are dropped
};

// ---------- Result types ----------

using DocId = uint64_t;

struct SearchResult {
    DocId doc_id;
    float score;
    bool operator>(const SearchResult& o) const { return score > o.score; }
};

} // namespace egbi
