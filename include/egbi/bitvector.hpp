#pragma once
// EGBI - Entropy-Gated Bit-Vector Indexing
// BitVector: fixed-length bit array with cross-platform popcount.

#include <cstdint>
#include <cstddef>
#include <vector>
#include <algorithm>
#include <cstring>
#include <stdexcept>

// ---- Cross-platform popcount intrinsic ----
#if defined(_MSC_VER)
  #include <intrin.h>
#endif

namespace egbi {

// Hardware-accelerated popcount with portable fallback.
inline uint32_t popcount64(uint64_t x) {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<uint32_t>(__builtin_popcountll(x));
#elif defined(_MSC_VER) && defined(_M_X64)
    return static_cast<uint32_t>(__popcnt64(x));
#elif defined(_MSC_VER) && defined(_M_IX86)
    return static_cast<uint32_t>(__popcnt(static_cast<uint32_t>(x)))
         + static_cast<uint32_t>(__popcnt(static_cast<uint32_t>(x >> 32)));
#else
    // Hamming-weight bit-twiddling fallback
    x -= (x >> 1) & 0x5555555555555555ULL;
    x  = (x & 0x3333333333333333ULL) + ((x >> 2) & 0x3333333333333333ULL);
    x  = (x + (x >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return static_cast<uint32_t>((x * 0x0101010101010101ULL) >> 56);
#endif
}

// ---------- BitVector ----------

class BitVector {
public:
    explicit BitVector(uint32_t num_bits)
        : num_bits_(num_bits)
        , data_((num_bits + 63) / 64, 0)
    {}

    BitVector() : num_bits_(0) {}

    // --- Bit manipulation ---

    void set_bit(uint32_t pos) {
        if (pos < num_bits_)
            data_[pos >> 6] |= (1ULL << (pos & 63));
    }

    void clear_bit(uint32_t pos) {
        if (pos < num_bits_)
            data_[pos >> 6] &= ~(1ULL << (pos & 63));
    }

    bool get_bit(uint32_t pos) const {
        if (pos >= num_bits_) return false;
        return (data_[pos >> 6] >> (pos & 63)) & 1;
    }

    void clear() {
        std::fill(data_.begin(), data_.end(), 0);
    }

    // --- Counting ---

    uint32_t popcount() const {
        uint32_t c = 0;
        for (auto w : data_) c += popcount64(w);
        return c;
    }

    // |A AND B|  (intersection bits set)
    static uint32_t and_popcount(const BitVector& a, const BitVector& b) {
        uint32_t c = 0;
        size_t n = std::min(a.data_.size(), b.data_.size());
        for (size_t i = 0; i < n; ++i)
            c += popcount64(a.data_[i] & b.data_[i]);
        return c;
    }

    // |A OR B|  (union bits set)
    static uint32_t or_popcount(const BitVector& a, const BitVector& b) {
        uint32_t c = 0;
        size_t n = std::max(a.data_.size(), b.data_.size());
        for (size_t i = 0; i < n; ++i) {
            uint64_t wa = (i < a.data_.size()) ? a.data_[i] : 0;
            uint64_t wb = (i < b.data_.size()) ? b.data_[i] : 0;
            c += popcount64(wa | wb);
        }
        return c;
    }

    // Bit-Jaccard:  |A & B| / |A | B|
    // Fused single-pass for cache friendliness.
    static float jaccard(const BitVector& a, const BitVector& b) {
        uint32_t and_cnt = 0, or_cnt = 0;
        size_t n = std::max(a.data_.size(), b.data_.size());
        for (size_t i = 0; i < n; ++i) {
            uint64_t wa = (i < a.data_.size()) ? a.data_[i] : 0;
            uint64_t wb = (i < b.data_.size()) ? b.data_[i] : 0;
            and_cnt += popcount64(wa & wb);
            or_cnt  += popcount64(wa | wb);
        }
        return (or_cnt == 0) ? 0.0f
             : static_cast<float>(and_cnt) / static_cast<float>(or_cnt);
    }

    // --- Accessors ---

    uint32_t                     num_bits() const { return num_bits_; }
    size_t                       word_count() const { return data_.size(); }
    const std::vector<uint64_t>& data() const { return data_; }
    std::vector<uint64_t>&       data()       { return data_; }

    // --- Binary serialization helpers ---

    // Write raw data to a byte buffer, appending.
    void serialize(std::vector<uint8_t>& buf) const {
        auto push32 = [&](uint32_t v) {
            buf.insert(buf.end(),
                reinterpret_cast<const uint8_t*>(&v),
                reinterpret_cast<const uint8_t*>(&v) + 4);
        };
        push32(num_bits_);
        push32(static_cast<uint32_t>(data_.size()));
        for (auto w : data_) {
            buf.insert(buf.end(),
                reinterpret_cast<const uint8_t*>(&w),
                reinterpret_cast<const uint8_t*>(&w) + 8);
        }
    }

    // Read from a byte pointer, advancing it. Returns false on underflow.
    bool deserialize(const uint8_t*& ptr, const uint8_t* end) {
        if (ptr + 8 > end) return false;
        uint32_t nb, nw;
        std::memcpy(&nb, ptr, 4); ptr += 4;
        std::memcpy(&nw, ptr, 4); ptr += 4;
        if (nw > 1000000) return false;            // sanity cap
        if (ptr + nw * 8 > end) return false;
        num_bits_ = nb;
        data_.resize(nw);
        for (uint32_t i = 0; i < nw; ++i) {
            std::memcpy(&data_[i], ptr, 8);
            ptr += 8;
        }
        return true;
    }

private:
    uint32_t              num_bits_;
    std::vector<uint64_t> data_;
};

} // namespace egbi
