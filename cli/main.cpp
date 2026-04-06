// EGBI-CLI — Command-line interface for Entropy-Gated Bit-Vector Indexing
//
// Commands:
//   build   <corpus.txt> <index.egbi>  [-k bits] [-a alpha] [-n ngram]
//   search  <index.egbi> <query>       [-t threshold] [-m max] [--corpus file]
//   eval    <corpus.txt>               [-k bits] [-a alpha] [-n ngram] [--pairs N]
//   bench   [--docs N] [--queries Q]   [-k bits] [-a alpha] [-n ngram]
//   info    <index.egbi>

#include <egbi/egbi.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

// ============================================================
//  Helpers
// ============================================================

static std::vector<std::string> load_lines(const std::string& path) {
    std::vector<std::string> lines;
    std::ifstream f(path);
    if (!f) {
        std::cerr << "Error: cannot open file: " << path << "\n";
        return lines;
    }
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty())
            lines.push_back(std::move(line));
    }
    return lines;
}

struct Timer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
    Timer() : start(Clock::now()) {}
    double elapsed_ms() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }
    double elapsed_us() const {
        return std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    }
};

// Parse optional flag value: --flag value  or  -f value
static bool flag_str(int argc, char** argv, const char* long_name, const char* short_name, std::string& out) {
    for (int i = 1; i < argc - 1; ++i) {
        if ((long_name && std::strcmp(argv[i], long_name) == 0) ||
            (short_name && std::strcmp(argv[i], short_name) == 0)) {
            out = argv[i + 1];
            return true;
        }
    }
    return false;
}

static float flag_float(int argc, char** argv, const char* ln, const char* sn, float def) {
    std::string s;
    if (flag_str(argc, argv, ln, sn, s)) return std::stof(s);
    return def;
}

static int flag_int(int argc, char** argv, const char* ln, const char* sn, int def) {
    std::string s;
    if (flag_str(argc, argv, ln, sn, s)) return std::stoi(s);
    return def;
}

static egbi::Config parse_config(int argc, char** argv) {
    egbi::Config cfg;
    cfg.bit_length = static_cast<uint32_t>(flag_int(argc, argv, "--bits", "-k", 1024));
    cfg.alpha      = flag_float(argc, argv, "--alpha", "-a", 2.0f);
    cfg.ngram_size = static_cast<uint32_t>(flag_int(argc, argv, "--ngram", "-n", 3));
    cfg.min_bits   = static_cast<uint32_t>(flag_int(argc, argv, "--min-bits", nullptr, 1));
    cfg.max_bits   = static_cast<uint32_t>(flag_int(argc, argv, "--max-bits", nullptr, 32));
    cfg.idf_floor  = flag_float(argc, argv, "--idf-floor", nullptr, 0.0f);
    return cfg;
}

// ============================================================
//  Ground truth: IDF-weighted set Jaccard
//
//  Architecture spec §3.3 defines:
//    S_EGBI(A,B) ≈ Σ_{t∈A∩B} b_t / Σ_{t∈A∪B} b_t
//  where b_t = ceil(α · IDF(t)) is the dynamic bit allocation.
//  The bit-Jaccard approximates this weighted set Jaccard,
//  NOT the multiset (TF-based) Jaccard.
// ============================================================

static float exact_weighted_jaccard(const std::string& a, const std::string& b,
                                     const egbi::EGBIIndex& idx) {
    auto& cfg = idx.config();
    auto ga = idx.extract_ngrams(a);
    auto gb = idx.extract_ngrams(b);

    std::unordered_set<std::string> sa(ga.begin(), ga.end());
    std::unordered_set<std::string> sb(gb.begin(), gb.end());

    // Compute b_t weight for each feature in the union
    float inter_w = 0.0f, union_w = 0.0f;
    std::unordered_set<std::string> all;
    all.insert(sa.begin(), sa.end());
    all.insert(sb.begin(), sb.end());

    for (const auto& t : all) {
        float idf = idx.get_idf(t);
        if (idf < cfg.idf_floor) continue;

        uint32_t b_t = static_cast<uint32_t>(std::ceil(cfg.alpha * idf));
        b_t = std::clamp(b_t, cfg.min_bits, cfg.max_bits);
        float w = static_cast<float>(b_t);

        bool in_a = sa.count(t) > 0;
        bool in_b = sb.count(t) > 0;
        if (in_a && in_b) inter_w += w;
        union_w += w;               // every feature in union contributes
    }

    return (union_w == 0.0f) ? 0.0f : inter_w / union_w;
}

// ============================================================
//  Command: build
// ============================================================

static int cmd_build(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: egbi-cli build <corpus.txt> <index.egbi> [options]\n";
        return 1;
    }
    std::string corpus_path = argv[2];
    std::string index_path  = argv[3];
    egbi::Config cfg = parse_config(argc, argv);

    auto docs = load_lines(corpus_path);
    if (docs.empty()) return 1;

    std::cout << "Building index from " << docs.size() << " documents...\n";
    std::cout << "  bit_length=" << cfg.bit_length
              << "  alpha=" << cfg.alpha
              << "  ngram=" << cfg.ngram_size << "\n";

    Timer t;
    egbi::EGBIIndex index(cfg);
    index.add_documents(docs);
    double build_ms = t.elapsed_ms();

    if (!index.save(index_path)) {
        std::cerr << "Error: failed to save index to " << index_path << "\n";
        return 1;
    }

    std::cout << "Done. " << index.size() << " documents indexed in "
              << std::fixed << std::setprecision(1) << build_ms << " ms\n";
    std::cout << "Index saved to: " << index_path << "\n";
    return 0;
}

// ============================================================
//  Command: search
// ============================================================

static int cmd_search(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "Usage: egbi-cli search <index.egbi> <query> [options]\n"
                  << "  -t <threshold>   minimum score (default: 0.1)\n"
                  << "  -m <max>         max results   (default: 10)\n"
                  << "  --corpus <file>  show original text alongside results\n";
        return 1;
    }
    std::string index_path = argv[2];
    std::string query      = argv[3];
    float threshold = flag_float(argc, argv, "--threshold", "-t", 0.1f);
    int max_results = flag_int(argc, argv, "--max", "-m", 10);
    std::string corpus_path;
    flag_str(argc, argv, "--corpus", nullptr, corpus_path);

    std::vector<std::string> corpus_lines;
    if (!corpus_path.empty())
        corpus_lines = load_lines(corpus_path);

    egbi::EGBIIndex index;
    if (!index.load(index_path)) {
        std::cerr << "Error: cannot load index from " << index_path << "\n";
        return 1;
    }

    Timer t;
    auto results = index.search(query, threshold, static_cast<size_t>(max_results));
    double search_us = t.elapsed_us();

    std::cout << "Query: \"" << query << "\"\n";
    std::cout << "Found " << results.size() << " results in "
              << std::fixed << std::setprecision(1) << search_us << " us\n";
    std::cout << std::string(64, '-') << "\n";

    for (auto& r : results) {
        std::cout << "  [" << std::setw(6) << r.doc_id << "]  score="
                  << std::fixed << std::setprecision(4) << r.score;
        if (r.doc_id < corpus_lines.size())
            std::cout << "  | " << corpus_lines[r.doc_id];
        std::cout << "\n";
    }
    return 0;
}

// ============================================================
//  Command: info
// ============================================================

static int cmd_info(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: egbi-cli info <index.egbi>\n";
        return 1;
    }
    egbi::EGBIIndex index;
    if (!index.load(argv[2])) {
        std::cerr << "Error: cannot load " << argv[2] << "\n";
        return 1;
    }
    auto& cfg = index.config();
    std::cout << "EGBI Index: " << argv[2] << "\n"
              << "  Documents:  " << index.size() << "\n"
              << "  bit_length: " << cfg.bit_length << "\n"
              << "  alpha:      " << cfg.alpha << "\n"
              << "  ngram_size: " << cfg.ngram_size << "\n"
              << "  min_bits:   " << cfg.min_bits << "\n"
              << "  max_bits:   " << cfg.max_bits << "\n"
              << "  idf_floor:  " << cfg.idf_floor << "\n"
              << "  Storage:    ~" << index.size() * (cfg.bit_length / 8)
              << " bytes (vectors only)\n";
    return 0;
}

// ============================================================
//  Standard IR Evaluation Metrics
// ============================================================

static double pearson_corr(const std::vector<float>& x, const std::vector<float>& y) {
    size_t n = x.size();
    if (n < 2) return 0.0;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (size_t i = 0; i < n; ++i) {
        sx  += x[i];  sy  += y[i];
        sxx += static_cast<double>(x[i]) * x[i];
        syy += static_cast<double>(y[i]) * y[i];
        sxy += static_cast<double>(x[i]) * y[i];
    }
    double num = n * sxy - sx * sy;
    double den = std::sqrt((n * sxx - sx * sx) * (n * syy - sy * sy));
    return (den < 1e-15) ? 0.0 : num / den;
}

static double spearman_corr(const std::vector<float>& x, const std::vector<float>& y) {
    size_t n = x.size();
    if (n < 2) return 0.0;
    auto to_ranks = [](const std::vector<float>& v) {
        size_t m = v.size();
        std::vector<size_t> ord(m);
        std::iota(ord.begin(), ord.end(), 0);
        std::sort(ord.begin(), ord.end(),
                  [&](size_t a, size_t b) { return v[a] > v[b]; });
        std::vector<float> ranks(m);
        for (size_t i = 0; i < m;) {
            size_t j = i;
            while (j < m && v[ord[j]] == v[ord[i]]) ++j;
            float avg_rank = static_cast<float>(i + j + 1) / 2.0f;
            for (size_t k = i; k < j; ++k) ranks[ord[k]] = avg_rank;
            i = j;
        }
        return ranks;
    };
    return pearson_corr(to_ranks(x), to_ranks(y));
}

static double dcg_at(const std::vector<float>& rel, int k) {
    double dcg = 0.0;
    for (int i = 0; i < std::min(k, static_cast<int>(rel.size())); ++i)
        dcg += rel[i] / std::log2(i + 2.0);
    return dcg;
}

static double ndcg_at(const std::vector<float>& pred, const std::vector<float>& ideal, int k) {
    double id = dcg_at(ideal, k);
    return (id < 1e-15) ? 1.0 : dcg_at(pred, k) / id;
}

static double avg_precision(const std::vector<bool>& is_rel) {
    double sum = 0.0;
    int hits = 0;
    for (size_t i = 0; i < is_rel.size(); ++i) {
        if (is_rel[i]) {
            ++hits;
            sum += static_cast<double>(hits) / (i + 1);
        }
    }
    return (hits == 0) ? 0.0 : sum / hits;
}

static double auc_pr(const std::vector<float>& scores, const std::vector<bool>& labels) {
    size_t n = scores.size();
    std::vector<size_t> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(),
              [&](size_t a, size_t b) { return scores[a] > scores[b]; });
    int total_pos = 0;
    for (auto b : labels) if (b) ++total_pos;
    if (total_pos == 0) return 0.0;
    double area = 0.0;
    int tp = 0;
    double prev_recall = 0.0;
    for (size_t i = 0; i < n; ++i) {
        if (labels[idx[i]]) ++tp;
        double prec   = static_cast<double>(tp) / (i + 1);
        double recall = static_cast<double>(tp) / total_pos;
        area += (recall - prev_recall) * prec;
        prev_recall = recall;
    }
    return area;
}

// ============================================================
//  Command: eval  — Standard IR Evaluation Benchmark
// ============================================================

static int cmd_eval(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: egbi-cli eval <corpus.txt> [options]\n"
                  << "  --pairs   N   random pairs for approximation quality (default: 500)\n"
                  << "  --queries N   queries for retrieval evaluation   (default: 20)\n"
                  << "  --seed    S   random seed (default: 42)\n";
        return 1;
    }
    std::string corpus_path = argv[2];
    egbi::Config cfg = parse_config(argc, argv);
    int num_pairs   = flag_int(argc, argv, "--pairs",   nullptr, 500);
    int num_queries = flag_int(argc, argv, "--queries", nullptr, 20);
    int seed        = flag_int(argc, argv, "--seed",    nullptr, 42);

    auto docs = load_lines(corpus_path);
    if (docs.size() < 2) {
        std::cerr << "Error: need at least 2 documents\n";
        return 1;
    }
    num_queries = std::min(num_queries, static_cast<int>(docs.size()));

    std::cout << "================================================================\n"
              << "                   EGBI Evaluation Report\n"
              << "================================================================\n\n"
              << "Config:  k=" << cfg.bit_length
              << "  alpha=" << cfg.alpha
              << "  ngram=" << cfg.ngram_size << "\n"
              << "Corpus:  " << docs.size() << " documents\n"
              << "Pairs:   " << num_pairs
              << "  |  Queries: " << num_queries
              << "  |  Seed: " << seed << "\n\n";

    Timer t_build;
    egbi::EGBIIndex index(cfg);
    index.add_documents(docs);
    double build_ms = t_build.elapsed_ms();
    std::cout << "Index built in " << std::fixed << std::setprecision(1)
              << build_ms << " ms\n\n";

    std::mt19937 rng(static_cast<uint32_t>(seed));

    // ── Section 1: Approximation Quality (random pairs) ──────
    std::cout << "--- 1. Approximation Quality (" << num_pairs << " pairs) ---\n";

    std::uniform_int_distribution<size_t> dist(0, docs.size() - 1);
    std::vector<float> exact_scores, approx_scores;
    exact_scores.reserve(static_cast<size_t>(num_pairs));
    approx_scores.reserve(static_cast<size_t>(num_pairs));
    double sum_abs = 0, sum_sq = 0, max_err = 0;

    for (int p = 0; p < num_pairs; ++p) {
        size_t i = dist(rng), j = dist(rng);
        while (j == i) j = dist(rng);

        float exact  = exact_weighted_jaccard(docs[i], docs[j], index);
        float approx = index.similarity(static_cast<egbi::DocId>(i),
                                         static_cast<egbi::DocId>(j));
        exact_scores.push_back(exact);
        approx_scores.push_back(approx);

        double err = std::fabs(approx - exact);
        sum_abs += err;
        sum_sq  += err * err;
        if (err > max_err) max_err = err;
    }

    double mae  = sum_abs / num_pairs;
    double rmse = std::sqrt(sum_sq / num_pairs);
    double pearson  = pearson_corr(exact_scores, approx_scores);
    double spearman = spearman_corr(exact_scores, approx_scores);

    std::cout << "  MAE:              " << std::fixed << std::setprecision(6) << mae << "\n"
              << "  RMSE:             " << rmse << "\n"
              << "  Max Error:        " << max_err << "\n"
              << "  Pearson  rho:     " << std::setprecision(4) << pearson << "\n"
              << "  Spearman rho:     " << spearman << "\n\n";

    // ── Section 2: Retrieval Quality (per-query IR metrics) ──
    std::cout << "--- 2. Retrieval Quality (" << num_queries << " queries) ---\n";

    std::vector<size_t> query_ids(docs.size());
    std::iota(query_ids.begin(), query_ids.end(), 0);
    std::shuffle(query_ids.begin(), query_ids.end(), rng);
    query_ids.resize(static_cast<size_t>(num_queries));

    constexpr int K_VALS[]  = {1, 5, 10};
    constexpr int NUM_K     = 3;
    double sum_pk[NUM_K]    = {};
    double sum_rk[NUM_K]    = {};
    double sum_ndcg[NUM_K]  = {};
    double sum_map = 0, sum_mrr = 0;
    std::vector<double> all_ndcg10;
    all_ndcg10.reserve(static_cast<size_t>(num_queries));

    for (auto q : query_ids) {
        // Ground truth: exact score to every other doc
        struct DS { size_t id; float score; };
        std::vector<DS> gt;
        gt.reserve(docs.size() - 1);
        for (size_t d = 0; d < docs.size(); ++d) {
            if (d == q) continue;
            gt.push_back({d, exact_weighted_jaccard(docs[q], docs[d], index)});
        }
        std::sort(gt.begin(), gt.end(),
                  [](const DS& a, const DS& b) { return a.score > b.score; });

        // EGBI ranking (exclude self)
        auto egbi_res = index.search(docs[q], 0.0f, docs.size());
        egbi_res.erase(
            std::remove_if(egbi_res.begin(), egbi_res.end(),
                           [q](const egbi::SearchResult& r) {
                               return static_cast<size_t>(r.doc_id) == q;
                           }),
            egbi_res.end());

        // Map doc_id → ground truth score
        std::unordered_map<size_t, float> gt_map;
        for (auto& g : gt) gt_map[g.id] = g.score;

        // Predicted relevance in EGBI rank order (for NDCG)
        std::vector<float> pred_rel, ideal_rel;
        ideal_rel.reserve(gt.size());
        for (auto& g : gt) ideal_rel.push_back(g.score);
        pred_rel.reserve(egbi_res.size());
        for (auto& r : egbi_res) {
            auto it = gt_map.find(static_cast<size_t>(r.doc_id));
            pred_rel.push_back(it != gt_map.end() ? it->second : 0.0f);
        }

        // NDCG@K
        for (int ki = 0; ki < NUM_K; ++ki)
            sum_ndcg[ki] += ndcg_at(pred_rel, ideal_rel, K_VALS[ki]);
        all_ndcg10.push_back(ndcg_at(pred_rel, ideal_rel, 10));

        // P@K / R@K  (relevant = ground truth top-K)
        for (int ki = 0; ki < NUM_K; ++ki) {
            int k = K_VALS[ki];
            std::unordered_set<size_t> gt_topk;
            for (int j = 0; j < std::min(k, static_cast<int>(gt.size())); ++j)
                gt_topk.insert(gt[static_cast<size_t>(j)].id);
            int hits = 0;
            for (int j = 0; j < std::min(k, static_cast<int>(egbi_res.size())); ++j)
                if (gt_topk.count(static_cast<size_t>(egbi_res[static_cast<size_t>(j)].doc_id)))
                    ++hits;
            int actual_k = std::min(k, static_cast<int>(gt.size()));
            sum_pk[ki] += (k > 0) ? static_cast<double>(hits) / k : 0.0;
            sum_rk[ki] += (actual_k > 0) ? static_cast<double>(hits) / actual_k : 0.0;
        }

        // MAP (relevant = ground truth top-10)
        {
            int rel_k = std::min(10, static_cast<int>(gt.size()));
            std::unordered_set<size_t> rel_set;
            for (int j = 0; j < rel_k; ++j) rel_set.insert(gt[static_cast<size_t>(j)].id);
            std::vector<bool> is_rel;
            is_rel.reserve(egbi_res.size());
            for (auto& r : egbi_res)
                is_rel.push_back(rel_set.count(static_cast<size_t>(r.doc_id)) > 0);
            sum_map += avg_precision(is_rel);
        }

        // MRR (first GT-top-1 in EGBI ranking)
        if (!gt.empty()) {
            size_t best_id = gt[0].id;
            for (size_t j = 0; j < egbi_res.size(); ++j) {
                if (static_cast<size_t>(egbi_res[j].doc_id) == best_id) {
                    sum_mrr += 1.0 / (j + 1);
                    break;
                }
            }
        }
    }

    double nq = static_cast<double>(num_queries);
    std::cout << std::setw(10) << ""
              << std::setw(9) << "P@1"  << std::setw(9) << "P@5"
              << std::setw(9) << "P@10" << std::setw(9) << "MAP"
              << std::setw(9) << "MRR"  << std::setw(10) << "NDCG@5"
              << std::setw(10) << "NDCG@10" << "\n";
    std::cout << "  Mean  " << std::fixed << std::setprecision(4)
              << std::setw(9) << sum_pk[0] / nq
              << std::setw(9) << sum_pk[1] / nq
              << std::setw(9) << sum_pk[2] / nq
              << std::setw(9) << sum_map / nq
              << std::setw(9) << sum_mrr / nq
              << std::setw(10) << sum_ndcg[1] / nq
              << std::setw(10) << sum_ndcg[2] / nq << "\n";
    double ndcg10_mean = sum_ndcg[2] / nq;
    double ndcg10_var = 0;
    for (auto v : all_ndcg10) ndcg10_var += (v - ndcg10_mean) * (v - ndcg10_mean);
    ndcg10_var /= nq;
    std::cout << "  R@1=" << std::setprecision(4) << sum_rk[0] / nq
              << "  R@5=" << sum_rk[1] / nq
              << "  R@10=" << sum_rk[2] / nq
              << "  NDCG@10 stdev=" << std::sqrt(ndcg10_var) << "\n\n";

    // ── Section 3: Classification (threshold sweep) ──────────
    std::cout << "--- 3. Classification (threshold sweep) ---\n";
    std::cout << "  Threshold  Precision  Recall     F1       Accuracy\n"
              << "  -------------------------------------------------------\n";

    const float thresholds[] = {0.05f, 0.10f, 0.15f, 0.20f, 0.30f, 0.40f, 0.50f};
    float best_f1 = 0, best_f1_thresh = 0;

    // Binary labels for AUC-PR (reference threshold = 0.20)
    const float ref_thresh = 0.20f;
    std::vector<bool> auc_labels;
    auc_labels.reserve(exact_scores.size());
    for (auto e : exact_scores) auc_labels.push_back(e >= ref_thresh);

    for (float th : thresholds) {
        int tp = 0, fp = 0, fn_v = 0, tn = 0;
        for (size_t i = 0; i < exact_scores.size(); ++i) {
            bool ep = (exact_scores[i]  >= th);
            bool ap = (approx_scores[i] >= th);
            if (ap && ep)  tp++;
            if (ap && !ep) fp++;
            if (!ap && ep) fn_v++;
            if (!ap && !ep) tn++;
        }
        float prec = (tp + fp > 0) ? static_cast<float>(tp) / (tp + fp) : 0;
        float rec  = (tp + fn_v > 0) ? static_cast<float>(tp) / (tp + fn_v) : 0;
        float f1   = (prec + rec > 0) ? 2 * prec * rec / (prec + rec) : 0;
        float acc  = static_cast<float>(tp + tn) / static_cast<float>(tp + fp + fn_v + tn);

        std::cout << "     " << std::fixed << std::setprecision(2) << th
                  << "    " << std::setprecision(4)
                  << std::setw(9) << prec
                  << std::setw(9) << rec
                  << std::setw(9) << f1
                  << std::setw(9) << acc << "\n";
        if (f1 > best_f1) { best_f1 = f1; best_f1_thresh = th; }
    }

    double auc = auc_pr(approx_scores, auc_labels);
    std::cout << "  -------------------------------------------------------\n"
              << "  Best F1: " << std::setprecision(4) << best_f1
              << " @ threshold=" << std::setprecision(2) << best_f1_thresh << "\n"
              << "  AUC-PR (ref>=" << ref_thresh << "): "
              << std::setprecision(4) << auc << "\n\n";

    // ── Section 4: Convergence (bit-length scaling) ──────────
    std::cout << "--- 4. Convergence (bit-length scaling) ---\n";
    std::cout << std::setw(8) << "k"
              << std::setw(12) << "MAE"
              << std::setw(12) << "Spearman"
              << std::setw(12) << "NDCG@10" << "\n"
              << "  -----------------------------------------\n";

    int conv_pairs   = std::min(num_pairs, 200);
    int conv_queries = std::min(num_queries, std::min(10, static_cast<int>(docs.size())));

    for (uint32_t k : {256u, 512u, 1024u, 2048u, 4096u}) {
        egbi::Config kcfg = cfg;
        kcfg.bit_length = k;
        egbi::EGBIIndex kidx(kcfg);
        kidx.add_documents(docs);

        // MAE + Spearman
        std::mt19937 rng2(static_cast<uint32_t>(seed));
        std::uniform_int_distribution<size_t> d2(0, docs.size() - 1);
        std::vector<float> ex, ap;
        ex.reserve(static_cast<size_t>(conv_pairs));
        ap.reserve(static_cast<size_t>(conv_pairs));
        double s_abs = 0;
        for (int p = 0; p < conv_pairs; ++p) {
            size_t i = d2(rng2), j = d2(rng2);
            while (j == i) j = d2(rng2);
            float e = exact_weighted_jaccard(docs[i], docs[j], kidx);
            float a = kidx.similarity(static_cast<egbi::DocId>(i),
                                       static_cast<egbi::DocId>(j));
            ex.push_back(e); ap.push_back(a);
            s_abs += std::fabs(a - e);
        }
        double k_mae      = s_abs / conv_pairs;
        double k_spearman = spearman_corr(ex, ap);

        // NDCG@10
        double k_ndcg = 0;
        std::vector<size_t> cq(docs.size());
        std::iota(cq.begin(), cq.end(), 0);
        std::shuffle(cq.begin(), cq.end(), rng2);
        for (int qi = 0; qi < conv_queries; ++qi) {
            size_t qd = cq[static_cast<size_t>(qi)];
            struct DS2 { size_t id; float score; };
            std::vector<DS2> gtv;
            for (size_t d = 0; d < docs.size(); ++d) {
                if (d == qd) continue;
                gtv.push_back({d, exact_weighted_jaccard(docs[qd], docs[d], kidx)});
            }
            std::sort(gtv.begin(), gtv.end(),
                      [](auto& a, auto& b) { return a.score > b.score; });

            auto res = kidx.search(docs[qd], 0.0f, docs.size());
            res.erase(std::remove_if(res.begin(), res.end(),
                      [qd](auto& r) { return static_cast<size_t>(r.doc_id) == qd; }),
                      res.end());

            std::unordered_map<size_t, float> gsm;
            for (auto& g : gtv) gsm[g.id] = g.score;
            std::vector<float> id_rel, pr_rel;
            for (auto& g : gtv) id_rel.push_back(g.score);
            for (auto& r : res) {
                auto it = gsm.find(static_cast<size_t>(r.doc_id));
                pr_rel.push_back(it != gsm.end() ? it->second : 0.0f);
            }
            k_ndcg += ndcg_at(pr_rel, id_rel, 10);
        }
        k_ndcg /= conv_queries;

        std::cout << std::setw(8) << k
                  << std::setw(12) << std::fixed << std::setprecision(6) << k_mae
                  << std::setw(12) << std::setprecision(4) << k_spearman
                  << std::setw(12) << k_ndcg << "\n";
    }

    std::cout << "\n--- 5. Worst Approximations (top 5) ---\n";
    struct PR { size_t i, j; float exact, approx, err; };
    std::vector<PR> worst;
    worst.reserve(exact_scores.size());
    // rebuild RNG to get same pairs
    std::mt19937 rng3(static_cast<uint32_t>(seed));
    std::uniform_int_distribution<size_t> d3(0, docs.size() - 1);
    for (size_t p = 0; p < exact_scores.size(); ++p) {
        size_t i = d3(rng3), j = d3(rng3);
        while (j == i) j = d3(rng3);
        worst.push_back({i, j, exact_scores[p], approx_scores[p],
                          std::fabs(exact_scores[p] - approx_scores[p])});
    }
    std::sort(worst.begin(), worst.end(),
              [](auto& a, auto& b) { return a.err > b.err; });
    for (int i = 0; i < std::min(5, static_cast<int>(worst.size())); ++i) {
        auto& w = worst[static_cast<size_t>(i)];
        std::cout << "  doc[" << w.i << "] vs doc[" << w.j << "]"
                  << "  exact=" << std::setprecision(4) << w.exact
                  << "  EGBI=" << w.approx
                  << "  err=" << w.err << "\n";
    }

    std::cout << "\nEvaluation complete.\n";
    return 0;
}

// ============================================================
//  Command: bench  — Scalability benchmark
// ============================================================

static std::string gen_random_doc(std::mt19937& rng, size_t avg_len) {
    std::normal_distribution<double> len_dist(static_cast<double>(avg_len),
                                               static_cast<double>(avg_len) * 0.3);
    size_t len = static_cast<size_t>(std::max(10.0, len_dist(rng)));
    std::uniform_int_distribution<int> char_dist(32, 126);
    std::string s;
    s.reserve(len);
    for (size_t i = 0; i < len; ++i)
        s.push_back(static_cast<char>(char_dist(rng)));
    return s;
}

static int cmd_bench(int argc, char** argv) {
    egbi::Config cfg = parse_config(argc, argv);
    int num_docs    = flag_int(argc, argv, "--docs",    nullptr, 10000);
    int num_queries = flag_int(argc, argv, "--queries", nullptr, 50);
    int doc_len     = flag_int(argc, argv, "--doc-len", nullptr, 200);
    int seed        = flag_int(argc, argv, "--seed",    nullptr, 42);

    std::cout << "=== EGBI Scalability Benchmark ===\n"
              << "Documents:  " << num_docs << "\n"
              << "Queries:    " << num_queries << "\n"
              << "Doc length: ~" << doc_len << " chars\n"
              << "bit_length: " << cfg.bit_length
              << "  alpha: " << cfg.alpha
              << "  ngram: " << cfg.ngram_size << "\n\n";

    std::mt19937 rng(static_cast<uint32_t>(seed));

    // Generate corpus
    std::cout << "Generating synthetic corpus... " << std::flush;
    std::vector<std::string> docs;
    docs.reserve(static_cast<size_t>(num_docs));
    for (int i = 0; i < num_docs; ++i)
        docs.push_back(gen_random_doc(rng, static_cast<size_t>(doc_len)));
    std::cout << "done.\n";

    // --- Benchmark: IDF build ---
    std::cout << "\n--- IDF Build ---\n";
    Timer t_idf;
    egbi::EGBIIndex index(cfg);
    index.build_idf(docs);
    double idf_ms = t_idf.elapsed_ms();
    std::cout << "  Time: " << std::fixed << std::setprecision(1) << idf_ms << " ms\n";

    // --- Benchmark: Index build ---
    std::cout << "\n--- Index Build ---\n";
    Timer t_idx;
    index.clear();
    index.add_documents(docs);
    double idx_ms = t_idx.elapsed_ms();
    std::cout << "  Time: " << std::fixed << std::setprecision(1) << idx_ms << " ms\n";
    std::cout << "  Throughput: " << std::fixed << std::setprecision(0)
              << (num_docs / (idx_ms / 1000.0)) << " docs/sec\n";

    // Storage estimate
    size_t vec_bytes = index.size() * (cfg.bit_length / 8 + 16);
    std::cout << "  Storage (vectors): " << vec_bytes / 1024 << " KB"
              << "  (" << (vec_bytes / index.size()) << " bytes/doc)\n";

    // --- Benchmark: Search latency ---
    std::cout << "\n--- Search Latency ---\n";
    std::vector<double> latencies;
    latencies.reserve(static_cast<size_t>(num_queries));

    for (int q = 0; q < num_queries; ++q) {
        std::string query = gen_random_doc(rng, static_cast<size_t>(doc_len) / 2);
        Timer t_q;
        auto results = index.search(query, 0.05f, 10);
        latencies.push_back(t_q.elapsed_us());
        (void)results;
    }

    std::sort(latencies.begin(), latencies.end());
    double avg = std::accumulate(latencies.begin(), latencies.end(), 0.0) / latencies.size();
    double p50 = latencies[latencies.size() / 2];
    double p95 = latencies[static_cast<size_t>(latencies.size() * 0.95)];
    double p99 = latencies[static_cast<size_t>(latencies.size() * 0.99)];

    std::cout << "  Avg:  " << std::fixed << std::setprecision(0) << avg << " us\n";
    std::cout << "  P50:  " << p50 << " us\n";
    std::cout << "  P95:  " << p95 << " us\n";
    std::cout << "  P99:  " << p99 << " us\n";
    std::cout << "  QPS:  " << std::fixed << std::setprecision(0)
              << (1000000.0 / avg) << " queries/sec\n";

    // --- Benchmark: Save / Load ---
    std::cout << "\n--- Serialization ---\n";
    const std::string bench_path = "bench_temp.egbi";
    Timer t_save;
    index.save(bench_path);
    double save_ms = t_save.elapsed_ms();
    std::cout << "  Save: " << std::fixed << std::setprecision(1) << save_ms << " ms\n";

    Timer t_load;
    egbi::EGBIIndex loaded;
    loaded.load(bench_path);
    double load_ms = t_load.elapsed_ms();
    std::cout << "  Load: " << std::fixed << std::setprecision(1) << load_ms << " ms\n";
    std::remove(bench_path.c_str());

    // --- Benchmark: Scalability across different k values ---
    std::cout << "\n--- Bit-Length Scalability ---\n";
    std::cout << std::setw(12) << "bit_length"
              << std::setw(14) << "build(ms)"
              << std::setw(14) << "search(us)"
              << std::setw(14) << "bytes/doc" << "\n";
    std::cout << std::string(54, '-') << "\n";

    for (uint32_t k : {256u, 512u, 1024u, 2048u, 4096u}) {
        egbi::Config kcfg = cfg;
        kcfg.bit_length = k;
        egbi::EGBIIndex kidx(kcfg);

        Timer tb;
        kidx.add_documents(docs);
        double bms = tb.elapsed_ms();

        std::string qtext = gen_random_doc(rng, static_cast<size_t>(doc_len) / 2);
        Timer ts;
        kidx.search(qtext, 0.05f, 10);
        double sus = ts.elapsed_us();

        std::cout << std::setw(12) << k
                  << std::setw(14) << std::fixed << std::setprecision(1) << bms
                  << std::setw(14) << std::fixed << std::setprecision(0) << sus
                  << std::setw(14) << (k / 8 + 16) << "\n";
    }

    // --- Benchmark: Document count scalability ---
    std::cout << "\n--- Document Count Scalability ---\n";
    std::cout << std::setw(12) << "num_docs"
              << std::setw(14) << "build(ms)"
              << std::setw(14) << "search(us)"
              << std::setw(14) << "QPS" << "\n";
    std::cout << std::string(54, '-') << "\n";

    for (int nd : {1000, 5000, 10000, 50000}) {
        if (nd > num_docs) {
            // Generate more docs if needed
            while (static_cast<int>(docs.size()) < nd)
                docs.push_back(gen_random_doc(rng, static_cast<size_t>(doc_len)));
        }
        std::vector<std::string> subset(docs.begin(), docs.begin() + nd);

        egbi::EGBIIndex sidx(cfg);
        Timer tb;
        sidx.add_documents(subset);
        double bms = tb.elapsed_ms();

        // Average over multiple queries
        double total_us = 0.0;
        int nq = 10;
        for (int qi = 0; qi < nq; ++qi) {
            std::string qtext = gen_random_doc(rng, static_cast<size_t>(doc_len) / 2);
            Timer ts;
            sidx.search(qtext, 0.05f, 10);
            total_us += ts.elapsed_us();
        }
        double avg_us = total_us / nq;

        std::cout << std::setw(12) << nd
                  << std::setw(14) << std::fixed << std::setprecision(1) << bms
                  << std::setw(14) << std::fixed << std::setprecision(0) << avg_us
                  << std::setw(14) << std::fixed << std::setprecision(0) << (1000000.0 / avg_us) << "\n";
    }

    std::cout << "\nBenchmark complete.\n";
    return 0;
}

// ============================================================
//  Help & main dispatch
// ============================================================

static void print_help() {
    std::cout <<
R"(EGBI-CLI — Entropy-Gated Bit-Vector Indexing

Usage: egbi-cli <command> [arguments] [options]

Commands:
  build  <corpus.txt> <index.egbi>   Build index from a text file (one doc per line)
  search <index.egbi> <query>        Fuzzy search against an index
  eval   <corpus.txt>                Evaluate approximation accuracy vs exact Jaccard
  bench  [options]                   Run scalability benchmarks (synthetic data)
  info   <index.egbi>                Show index metadata

Global Options:
  -k, --bits   <N>        Bit-vector length (default: 1024)
  -a, --alpha  <F>        Dynamic bit allocation scaling (default: 2.0)
  -n, --ngram  <N>        N-gram size (default: 3)
  --min-bits   <N>        Min bits per feature (default: 1)
  --max-bits   <N>        Max bits per feature (default: 32)
  --idf-floor  <F>        Drop features with IDF below this (default: 0.0)

Search Options:
  -t, --threshold <F>     Minimum similarity score (default: 0.1)
  -m, --max       <N>     Maximum results returned (default: 10)
  --corpus        <file>  Corpus file to display original text in results

Eval Options:
  --pairs   <N>           Random pairs for approximation quality (default: 500)
  --queries <N>           Queries for retrieval evaluation (default: 20)
  --seed    <N>           Random seed (default: 42)

Bench Options:
  --docs    <N>           Number of synthetic documents (default: 10000)
  --queries <N>           Number of search queries (default: 50)
  --doc-len <N>           Average document length in chars (default: 200)
  --seed    <N>           Random seed (default: 42)

Examples:
  egbi-cli build corpus.txt my.egbi -k 2048 -a 2.5
  egbi-cli search my.egbi "quick brown fox" -t 0.2 --corpus corpus.txt
  egbi-cli eval corpus.txt --pairs 1000 --queries 30 -k 2048
  egbi-cli bench --docs 50000 --queries 100 -k 1024
  egbi-cli info my.egbi
)";
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_help();
        return 1;
    }

    std::string cmd = argv[1];

    if (cmd == "build")       return cmd_build(argc, argv);
    if (cmd == "search")      return cmd_search(argc, argv);
    if (cmd == "eval")        return cmd_eval(argc, argv);
    if (cmd == "bench")       return cmd_bench(argc, argv);
    if (cmd == "info")        return cmd_info(argc, argv);
    if (cmd == "help" || cmd == "--help" || cmd == "-h") {
        print_help();
        return 0;
    }

    std::cerr << "Unknown command: " << cmd << "\n";
    print_help();
    return 1;
}
