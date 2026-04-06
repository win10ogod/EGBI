#pragma once
// EGBI - Entropy-Gated Bit-Vector Indexing
// Core index: IDF computation, dynamic bit allocation, fuzzy search.

#include "types.hpp"
#include "bitvector.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace egbi {

// ======== FNV-1a hash (64-bit) with seed ========

inline uint64_t fnv1a(const char* data, size_t len, uint64_t seed = 0) {
    uint64_t h = 14695981039346656037ULL ^ seed;
    for (size_t i = 0; i < len; ++i) {
        h ^= static_cast<uint64_t>(static_cast<uint8_t>(data[i]));
        h *= 1099511628211ULL;
    }
    return h;
}

inline uint64_t fnv1a(const std::string& s, uint64_t seed = 0) {
    return fnv1a(s.data(), s.size(), seed);
}

// ======== EGBIIndex ========

class EGBIIndex {
public:
    explicit EGBIIndex(const Config& cfg = {}) : cfg_(cfg) {}

    // ------ Feature extraction ------

    // Character-level n-gram (byte window). Works with any encoding.
    std::vector<std::string> extract_ngrams(const std::string& text) const {
        std::vector<std::string> grams;
        if (text.size() < cfg_.ngram_size) {
            if (!text.empty()) grams.push_back(text);
            return grams;
        }
        grams.reserve(text.size() - cfg_.ngram_size + 1);
        for (size_t i = 0; i + cfg_.ngram_size <= text.size(); ++i)
            grams.emplace_back(text, i, cfg_.ngram_size);
        return grams;
    }

    // ------ IDF / entropy statistics ------

    // Build IDF table from a corpus. Call before adding documents for best accuracy.
    void build_idf(const std::vector<std::string>& documents) {
        doc_count_ = documents.size();
        df_.clear();
        for (const auto& doc : documents) {
            auto grams = extract_ngrams(doc);
            std::unordered_set<std::string> seen;
            for (const auto& g : grams) {
                if (seen.insert(g).second)
                    df_[g]++;
            }
        }
    }

    // IDF for a single term: log((N+1) / (df+1))
    float get_idf(const std::string& term) const {
        float N = static_cast<float>(doc_count_ > 0 ? doc_count_ : 1);
        auto it = df_.find(term);
        float df = (it != df_.end()) ? static_cast<float>(it->second) : 0.0f;
        return std::log((N + 1.0f) / (df + 1.0f));
    }

    // ------ BitVector construction ------

    // Build a bit vector for arbitrary text.
    BitVector build_vector(const std::string& text) const {
        BitVector v(cfg_.bit_length);
        auto grams = extract_ngrams(text);
        for (const auto& g : grams) {
            float idf = get_idf(g);
            if (idf < cfg_.idf_floor) continue;  // drop low-entropy noise

            uint32_t b_t = static_cast<uint32_t>(std::ceil(cfg_.alpha * idf));
            b_t = std::clamp(b_t, cfg_.min_bits, cfg_.max_bits);

            for (uint32_t i = 0; i < b_t; ++i) {
                // Independent hash per slot: seed = i * large odd prime
                uint64_t h = fnv1a(g, static_cast<uint64_t>(i) * 2654435761ULL);
                v.set_bit(static_cast<uint32_t>(h % cfg_.bit_length));
            }
        }
        return v;
    }

    // Build from pre-tokenised input (word-level n-grams, custom tokenizer, etc.)
    BitVector build_vector_from_tokens(const std::vector<std::string>& tokens) const {
        BitVector v(cfg_.bit_length);
        for (const auto& tok : tokens) {
            float idf = get_idf(tok);
            if (idf < cfg_.idf_floor) continue;

            uint32_t b_t = static_cast<uint32_t>(std::ceil(cfg_.alpha * idf));
            b_t = std::clamp(b_t, cfg_.min_bits, cfg_.max_bits);

            for (uint32_t i = 0; i < b_t; ++i) {
                uint64_t h = fnv1a(tok, static_cast<uint64_t>(i) * 2654435761ULL);
                v.set_bit(static_cast<uint32_t>(h % cfg_.bit_length));
            }
        }
        return v;
    }

    // ------ Document management ------

    // Add one document, returns its DocId.
    DocId add_document(const std::string& text) {
        DocId id = static_cast<DocId>(vectors_.size());
        vectors_.push_back(build_vector(text));
        return id;
    }

    // Add a pre-built vector directly.
    DocId add_vector(BitVector&& v) {
        DocId id = static_cast<DocId>(vectors_.size());
        vectors_.push_back(std::move(v));
        return id;
    }

    // Batch: build IDF then index all documents at once.
    void add_documents(const std::vector<std::string>& documents) {
        build_idf(documents);
        vectors_.clear();
        vectors_.reserve(documents.size());
        for (const auto& doc : documents)
            vectors_.push_back(build_vector(doc));
    }

    void clear() {
        vectors_.clear();
        df_.clear();
        doc_count_ = 0;
    }

    // ------ Search ------

    // Fuzzy search: returns all documents with Jaccard >= threshold, sorted by score descending.
    std::vector<SearchResult> search(
        const std::string& query,
        float threshold   = 0.1f,
        size_t max_results = 0
    ) const {
        BitVector vq = build_vector(query);
        return search_by_vector(vq, threshold, max_results);
    }

    // Search with a pre-built query vector.
    std::vector<SearchResult> search_by_vector(
        const BitVector& vq,
        float threshold   = 0.1f,
        size_t max_results = 0
    ) const {
        std::vector<SearchResult> results;
        for (size_t i = 0; i < vectors_.size(); ++i) {
            float s = BitVector::jaccard(vq, vectors_[i]);
            if (s >= threshold)
                results.push_back({static_cast<DocId>(i), s});
        }
        std::sort(results.begin(), results.end(),
                  [](const SearchResult& a, const SearchResult& b) {
                      return a.score > b.score;
                  });
        if (max_results > 0 && results.size() > max_results)
            results.resize(max_results);
        return results;
    }

    // Pairwise similarity between two indexed documents.
    float similarity(DocId a, DocId b) const {
        if (a >= vectors_.size() || b >= vectors_.size()) return 0.0f;
        return BitVector::jaccard(vectors_[a], vectors_[b]);
    }

    // Similarity between query text and an indexed document.
    float similarity(const std::string& query, DocId doc) const {
        if (doc >= vectors_.size()) return 0.0f;
        return BitVector::jaccard(build_vector(query), vectors_[doc]);
    }

    // ------ Accessors ------

    size_t              size()    const { return vectors_.size(); }
    const Config&       config()  const { return cfg_; }
    const std::vector<BitVector>& vectors() const { return vectors_; }

    // ------ Persistence (binary) ------

    // File format:  MAGIC(4) VERSION(4) Config(20) doc_count(8)
    //               df_size(4) [strlen(4) chars(n) count(4)]...
    //               vec_count(4) [BitVector serialized]...
    static constexpr uint32_t MAGIC   = 0x49424745; // "EGBI" little-endian
    static constexpr uint32_t VERSION = 1;

    bool save(const std::string& path) const {
        std::vector<uint8_t> buf;
        buf.reserve(64 + vectors_.size() * (cfg_.bit_length / 8 + 16));

        auto push = [&](const void* p, size_t n) {
            buf.insert(buf.end(),
                static_cast<const uint8_t*>(p),
                static_cast<const uint8_t*>(p) + n);
        };
        auto push32 = [&](uint32_t v) { push(&v, 4); };
        auto push64 = [&](uint64_t v) { push(&v, 8); };
        auto pushf  = [&](float v)    { push(&v, 4); };

        push32(MAGIC);
        push32(VERSION);

        // Config
        push32(cfg_.bit_length);
        pushf(cfg_.alpha);
        push32(cfg_.ngram_size);
        push32(cfg_.min_bits);
        push32(cfg_.max_bits);
        pushf(cfg_.idf_floor);

        // IDF table
        push64(doc_count_);
        push32(static_cast<uint32_t>(df_.size()));
        for (const auto& [term, cnt] : df_) {
            push32(static_cast<uint32_t>(term.size()));
            push(term.data(), term.size());
            push32(cnt);
        }

        // Vectors
        push32(static_cast<uint32_t>(vectors_.size()));
        for (const auto& v : vectors_)
            v.serialize(buf);

        std::ofstream f(path, std::ios::binary);
        if (!f) return false;
        f.write(reinterpret_cast<const char*>(buf.data()),
                static_cast<std::streamsize>(buf.size()));
        return f.good();
    }

    bool load(const std::string& path) {
        std::ifstream f(path, std::ios::binary | std::ios::ate);
        if (!f) return false;
        auto sz = f.tellg();
        if (sz <= 0) return false;
        f.seekg(0);
        std::vector<uint8_t> buf(static_cast<size_t>(sz));
        f.read(reinterpret_cast<char*>(buf.data()), sz);
        if (!f.good()) return false;

        const uint8_t* ptr = buf.data();
        const uint8_t* end = ptr + buf.size();

        auto read32 = [&](uint32_t& v) -> bool {
            if (ptr + 4 > end) return false;
            std::memcpy(&v, ptr, 4); ptr += 4; return true;
        };
        auto read64 = [&](uint64_t& v) -> bool {
            if (ptr + 8 > end) return false;
            std::memcpy(&v, ptr, 8); ptr += 8; return true;
        };
        auto readf = [&](float& v) -> bool {
            if (ptr + 4 > end) return false;
            std::memcpy(&v, ptr, 4); ptr += 4; return true;
        };

        uint32_t magic, ver;
        if (!read32(magic) || magic != MAGIC) return false;
        if (!read32(ver)   || ver   != VERSION) return false;

        if (!read32(cfg_.bit_length)) return false;
        if (!readf(cfg_.alpha))       return false;
        if (!read32(cfg_.ngram_size)) return false;
        if (!read32(cfg_.min_bits))   return false;
        if (!read32(cfg_.max_bits))   return false;
        if (!readf(cfg_.idf_floor))   return false;

        if (!read64(doc_count_)) return false;
        uint32_t df_size;
        if (!read32(df_size)) return false;
        if (df_size > 100000000) return false;  // sanity
        df_.clear();
        df_.reserve(df_size);
        for (uint32_t i = 0; i < df_size; ++i) {
            uint32_t slen, cnt;
            if (!read32(slen)) return false;
            if (slen > 10000 || ptr + slen > end) return false;
            std::string term(reinterpret_cast<const char*>(ptr), slen);
            ptr += slen;
            if (!read32(cnt)) return false;
            df_[std::move(term)] = cnt;
        }

        uint32_t vec_count;
        if (!read32(vec_count)) return false;
        if (vec_count > 100000000) return false;
        vectors_.clear();
        vectors_.resize(vec_count);
        for (uint32_t i = 0; i < vec_count; ++i) {
            if (!vectors_[i].deserialize(ptr, end)) return false;
        }
        return true;
    }

private:
    Config                                     cfg_;
    size_t                                     doc_count_ = 0;
    std::unordered_map<std::string, uint32_t>  df_;       // document-frequency table
    std::vector<BitVector>                     vectors_;  // one per document
};

} // namespace egbi
