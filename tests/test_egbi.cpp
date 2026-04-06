// EGBI - Test Suite
// Minimal self-contained tests (no external framework required).

#include <egbi/egbi.hpp>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static int tests_run    = 0;
static int tests_passed = 0;

#define TEST(name)                                          \
    static void test_##name();                              \
    struct Register_##name {                                \
        Register_##name() { test_##name(); }                \
    } reg_##name;                                           \
    static void test_##name()

#define EXPECT(cond)                                        \
    do {                                                    \
        ++tests_run;                                        \
        if (cond) { ++tests_passed; }                       \
        else {                                              \
            std::fprintf(stderr, "  FAIL: %s  (%s:%d)\n",  \
                #cond, __FILE__, __LINE__);                 \
        }                                                   \
    } while (0)

#define EXPECT_NEAR(a, b, eps) EXPECT(std::fabs((a) - (b)) < (eps))

// ===========================================================
//  BitVector tests
// ===========================================================

TEST(bitvector_basic) {
    egbi::BitVector v(256);
    EXPECT(v.num_bits() == 256);
    EXPECT(v.popcount() == 0);

    v.set_bit(0);
    v.set_bit(63);
    v.set_bit(64);
    v.set_bit(255);
    EXPECT(v.popcount() == 4);
    EXPECT(v.get_bit(0) == true);
    EXPECT(v.get_bit(1) == false);
    EXPECT(v.get_bit(63) == true);
    EXPECT(v.get_bit(64) == true);
    EXPECT(v.get_bit(255) == true);

    v.clear_bit(0);
    EXPECT(v.get_bit(0) == false);
    EXPECT(v.popcount() == 3);
}

TEST(bitvector_jaccard) {
    egbi::BitVector a(128), b(128);
    // Identical vectors -> Jaccard = 1.0
    a.set_bit(10); a.set_bit(20); a.set_bit(30);
    b.set_bit(10); b.set_bit(20); b.set_bit(30);
    EXPECT_NEAR(egbi::BitVector::jaccard(a, b), 1.0f, 0.001f);

    // Disjoint vectors -> Jaccard = 0.0
    egbi::BitVector c(128);
    c.set_bit(40); c.set_bit(50);
    EXPECT_NEAR(egbi::BitVector::jaccard(a, c), 0.0f, 0.001f);

    // Partial overlap
    egbi::BitVector d(128);
    d.set_bit(10); d.set_bit(20); d.set_bit(40); // overlap: 10,20; union: 10,20,30,40
    EXPECT_NEAR(egbi::BitVector::jaccard(a, d), 2.0f / 4.0f, 0.001f);
}

TEST(bitvector_empty) {
    egbi::BitVector a(64), b(64);
    // Both empty -> 0.0 (not NaN)
    EXPECT_NEAR(egbi::BitVector::jaccard(a, b), 0.0f, 0.001f);
}

TEST(bitvector_serialize) {
    egbi::BitVector v(512);
    v.set_bit(0); v.set_bit(100); v.set_bit(511);

    std::vector<uint8_t> buf;
    v.serialize(buf);

    const uint8_t* ptr = buf.data();
    const uint8_t* end = ptr + buf.size();
    egbi::BitVector v2;
    EXPECT(v2.deserialize(ptr, end));
    EXPECT(v2.num_bits() == 512);
    EXPECT(v2.get_bit(0) == true);
    EXPECT(v2.get_bit(100) == true);
    EXPECT(v2.get_bit(511) == true);
    EXPECT(v2.get_bit(1) == false);
    EXPECT(v2.popcount() == 3);
}

// ===========================================================
//  Index tests
// ===========================================================

TEST(ngram_extraction) {
    egbi::EGBIIndex idx;
    auto grams = idx.extract_ngrams("abcde");
    // trigrams: abc, bcd, cde
    EXPECT(grams.size() == 3);
    EXPECT(grams[0] == "abc");
    EXPECT(grams[1] == "bcd");
    EXPECT(grams[2] == "cde");

    // Shorter than n
    auto g2 = idx.extract_ngrams("ab");
    EXPECT(g2.size() == 1);
    EXPECT(g2[0] == "ab");
}

TEST(identical_docs_score_one) {
    std::vector<std::string> docs = {
        "hello world",
        "hello world",
    };
    egbi::EGBIIndex idx;
    idx.add_documents(docs);
    float s = idx.similarity(0, 1);
    EXPECT_NEAR(s, 1.0f, 0.001f);
}

TEST(different_docs_low_score) {
    std::vector<std::string> docs = {
        "the quick brown fox jumps over the lazy dog",
        "quantum computing with superconducting qubits",
    };
    egbi::EGBIIndex idx;
    idx.add_documents(docs);
    float s = idx.similarity(0, 1);
    EXPECT(s < 0.3f);  // very different -> low similarity
}

TEST(similar_docs_moderate_score) {
    std::vector<std::string> docs = {
        "the quick brown fox jumps over the lazy dog",
        "the quick brown fox leaps over a sleepy dog",
    };
    egbi::EGBIIndex idx;
    idx.add_documents(docs);
    float s = idx.similarity(0, 1);
    EXPECT(s > 0.3f);  // similar sentences -> moderate-high similarity
}

TEST(search_returns_sorted) {
    std::vector<std::string> docs = {
        "alpha beta gamma",
        "alpha beta delta",
        "epsilon zeta eta",
    };
    egbi::EGBIIndex idx;
    idx.add_documents(docs);

    auto results = idx.search("alpha beta gamma", 0.01f);
    EXPECT(results.size() >= 1);
    // First result should be the most similar
    EXPECT(results[0].doc_id == 0);
    // Scores should be descending
    for (size_t i = 1; i < results.size(); ++i)
        EXPECT(results[i - 1].score >= results[i].score);
}

TEST(search_max_results) {
    std::vector<std::string> docs;
    for (int i = 0; i < 20; ++i)
        docs.push_back("document number " + std::to_string(i) + " with content");
    egbi::EGBIIndex idx;
    idx.add_documents(docs);

    auto results = idx.search("document number", 0.01f, 5);
    EXPECT(results.size() <= 5);
}

TEST(save_load_roundtrip) {
    std::vector<std::string> docs = {
        "hello world foo bar",
        "baz qux quux corge",
    };
    egbi::EGBIIndex idx;
    idx.add_documents(docs);

    const std::string path = "test_roundtrip.egbi";
    EXPECT(idx.save(path));

    egbi::EGBIIndex loaded;
    EXPECT(loaded.load(path));
    EXPECT(loaded.size() == 2);
    // Similarity should be identical after load
    EXPECT_NEAR(loaded.similarity(0, 1), idx.similarity(0, 1), 0.001f);

    // Clean up
    std::remove(path.c_str());
}

TEST(entropy_gating) {
    // A high-entropy (rare) term should contribute more bits.
    // Build a corpus where "aaa" is common and "xyz" is rare.
    std::vector<std::string> docs;
    for (int i = 0; i < 100; ++i)
        docs.push_back("aaa common text " + std::to_string(i));
    docs.push_back("xyz unique rare term");

    egbi::EGBIIndex idx;
    idx.build_idf(docs);

    float idf_common = idx.get_idf("aaa");
    float idf_rare   = idx.get_idf("xyz");
    EXPECT(idf_rare > idf_common);  // rare term has higher IDF
}

TEST(build_vector_from_tokens) {
    egbi::EGBIIndex idx;
    std::vector<std::string> docs = {"hello world", "foo bar"};
    idx.build_idf(docs);

    std::vector<std::string> tokens = {"hel", "ell", "llo"};
    auto v = idx.build_vector_from_tokens(tokens);
    EXPECT(v.popcount() > 0);
}

// ===========================================================
// popcount correctness
// ===========================================================

TEST(popcount64_known_values) {
    EXPECT(egbi::popcount64(0) == 0);
    EXPECT(egbi::popcount64(1) == 1);
    EXPECT(egbi::popcount64(0xFFFFFFFFFFFFFFFFULL) == 64);
    EXPECT(egbi::popcount64(0xAAAAAAAAAAAAAAAAULL) == 32);
    EXPECT(egbi::popcount64(0x8000000000000001ULL) == 2);
}

// ===========================================================

int main() {
    std::printf("\n=== EGBI Test Suite ===\n\n");
    // Tests already executed via static registration.
    std::printf("\n%d / %d tests passed.\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
