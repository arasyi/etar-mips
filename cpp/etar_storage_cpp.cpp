#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

extern "C" {

struct EtarStorageStatus {
    int32_t candidates;
    int32_t results;
};

struct EtarStorageProfile {
    float scan_ms;
    float top_r_ms;
    float rerank_ms;
    float total_native_ms;
    float block_prune_survival;
    float block_prune_used;
    float block_prune_fallback;
};

using Clock = std::chrono::high_resolution_clock;

static float elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<float, std::milli>(b - a).count();
}

static int8_t quantize_value(float x, float scale) {
    const float raw = scale > 0.0f ? x / scale : 0.0f;
    long rounded = std::lrint(raw);
    if (rounded > 127) rounded = 127;
    if (rounded < -127) rounded = -127;
    return static_cast<int8_t>(rounded);
}

static float exact_dot(const float *x, const float *q, int64_t row, int64_t d) {
    const float *xr = x + row * d;
    float acc = 0.0f;
    for (int64_t j = 0; j < d; ++j) acc += xr[j] * q[j];
    return acc;
}

static float quantized_dot_col(const std::vector<int8_t> &codes_col, const std::vector<float> &scales, const float *q, int64_t row, int64_t d, int64_t capacity) {
    float acc = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
        acc += static_cast<float>(codes_col[static_cast<size_t>(j * capacity + row)]) * q[j];
    }
    return scales[static_cast<size_t>(row)] * acc;
}

static float quantized_dot_row(const std::vector<int8_t> &codes_row, const std::vector<float> &scales, const float *q, int64_t row, int64_t d) {
    const int8_t *xr = codes_row.data() + row * d;
    float acc = 0.0f;
    for (int64_t j = 0; j < d; ++j) {
        acc += static_cast<float>(xr[j]) * q[j];
    }
    return scales[static_cast<size_t>(row)] * acc;
}

static float quantized_dot_col_q8(
    const std::vector<int8_t> &codes_col,
    const std::vector<float> &scales,
    const int8_t *q_codes,
    float q_scale,
    int64_t row,
    int64_t d,
    int64_t capacity
) {
    int32_t acc = 0;
    for (int64_t j = 0; j < d; ++j) {
        acc += static_cast<int32_t>(codes_col[static_cast<size_t>(j * capacity + row)]) *
            static_cast<int32_t>(q_codes[j]);
    }
    return scales[static_cast<size_t>(row)] * q_scale * static_cast<float>(acc);
}

static float quantized_dot_row_q8(
    const std::vector<int8_t> &codes_row,
    const std::vector<float> &scales,
    const int8_t *q_codes,
    float q_scale,
    int64_t row,
    int64_t d
) {
    const int8_t *xr = codes_row.data() + row * d;
    int32_t acc = 0;
    for (int64_t j = 0; j < d; ++j) {
        acc += static_cast<int32_t>(xr[j]) * static_cast<int32_t>(q_codes[j]);
    }
    return scales[static_cast<size_t>(row)] * q_scale * static_cast<float>(acc);
}

static inline void add_i8_column_scaled_i32(int32_t *scores, const int8_t *col, int64_t len, int32_t qj) {
    if (qj == 0) return;
    int64_t off = 0;
#if defined(__AVX512F__) && defined(__AVX512BW__)
    const __m512i q32 = _mm512_set1_epi32(qj);
    for (; off + 63 < len; off += 64) {
        const __m128i c0_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off));
        const __m128i c1_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off + 16));
        const __m128i c2_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off + 32));
        const __m128i c3_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off + 48));

        __m512i a0 = _mm512_loadu_si512(reinterpret_cast<const void *>(scores + off));
        __m512i a1 = _mm512_loadu_si512(reinterpret_cast<const void *>(scores + off + 16));
        __m512i a2 = _mm512_loadu_si512(reinterpret_cast<const void *>(scores + off + 32));
        __m512i a3 = _mm512_loadu_si512(reinterpret_cast<const void *>(scores + off + 48));

        a0 = _mm512_add_epi32(a0, _mm512_mullo_epi32(_mm512_cvtepi8_epi32(c0_bytes), q32));
        a1 = _mm512_add_epi32(a1, _mm512_mullo_epi32(_mm512_cvtepi8_epi32(c1_bytes), q32));
        a2 = _mm512_add_epi32(a2, _mm512_mullo_epi32(_mm512_cvtepi8_epi32(c2_bytes), q32));
        a3 = _mm512_add_epi32(a3, _mm512_mullo_epi32(_mm512_cvtepi8_epi32(c3_bytes), q32));

        _mm512_storeu_si512(reinterpret_cast<void *>(scores + off), a0);
        _mm512_storeu_si512(reinterpret_cast<void *>(scores + off + 16), a1);
        _mm512_storeu_si512(reinterpret_cast<void *>(scores + off + 32), a2);
        _mm512_storeu_si512(reinterpret_cast<void *>(scores + off + 48), a3);
    }
    for (; off + 15 < len; off += 16) {
        const __m128i c_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off));
        __m512i a = _mm512_loadu_si512(reinterpret_cast<const void *>(scores + off));
        a = _mm512_add_epi32(a, _mm512_mullo_epi32(_mm512_cvtepi8_epi32(c_bytes), q32));
        _mm512_storeu_si512(reinterpret_cast<void *>(scores + off), a);
    }
#elif defined(__AVX2__)
    const __m256i q16 = _mm256_set1_epi16(static_cast<int16_t>(qj));
    for (; off + 31 < len; off += 32) {
        const __m128i c0_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off));
        const __m128i c1_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off + 16));
        const __m256i c0 = _mm256_cvtepi8_epi16(c0_bytes);
        const __m256i c1 = _mm256_cvtepi8_epi16(c1_bytes);
        const __m256i p0 = _mm256_mullo_epi16(c0, q16);
        const __m256i p1 = _mm256_mullo_epi16(c1, q16);

        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scores + off));
        __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scores + off + 8));
        __m256i a2 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scores + off + 16));
        __m256i a3 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scores + off + 24));

        a0 = _mm256_add_epi32(a0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p0)));
        a1 = _mm256_add_epi32(a1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p0, 1)));
        a2 = _mm256_add_epi32(a2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p1)));
        a3 = _mm256_add_epi32(a3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p1, 1)));

        _mm256_storeu_si256(reinterpret_cast<__m256i *>(scores + off), a0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(scores + off + 8), a1);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(scores + off + 16), a2);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(scores + off + 24), a3);
    }
    for (; off + 15 < len; off += 16) {
        const __m128i c_bytes = _mm_loadu_si128(reinterpret_cast<const __m128i *>(col + off));
        const __m256i c = _mm256_cvtepi8_epi16(c_bytes);
        const __m256i p = _mm256_mullo_epi16(c, q16);
        __m256i a0 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scores + off));
        __m256i a1 = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(scores + off + 8));
        a0 = _mm256_add_epi32(a0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(p)));
        a1 = _mm256_add_epi32(a1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(p, 1)));
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(scores + off), a0);
        _mm256_storeu_si256(reinterpret_cast<__m256i *>(scores + off + 8), a1);
    }
#endif
    for (; off < len; ++off) {
        scores[off] += qj * static_cast<int32_t>(col[off]);
    }
}

static bool better_desc(const std::pair<float, int64_t> &a, const std::pair<float, int64_t> &b) {
    if (a.first == b.first) return a.second < b.second;
    return a.first > b.first;
}

static bool worse_for_min_heap(const std::pair<float, int64_t> &a, const std::pair<float, int64_t> &b) {
    if (a.first == b.first) return a.second > b.second;
    return a.first > b.first;
}

struct EtarStorageIndex {
    int64_t n = 0;
    int64_t capacity = 0;
    int64_t d = 0;
    int64_t live_count = 0;
    double growth_factor = 2.0;
    bool store_residuals = false;
    bool store_float32 = true;
    bool store_codes_row = false;
    std::vector<float> x;
    std::vector<int8_t> codes_col;
    std::vector<int8_t> codes_row;
    std::vector<float> scales;
    std::vector<float> norms;
    std::vector<float> residuals;
    std::vector<uint8_t> alive;
    std::vector<uint32_t> row_ids;
    std::unordered_map<uint32_t, int64_t> id_to_phys;
    int64_t next_row_id = 0;

    static constexpr uint32_t invalid_row_id() {
        return std::numeric_limits<uint32_t>::max();
    }

    static bool valid_row_id(int64_t row_id) {
        return row_id >= 0 && row_id < static_cast<int64_t>(invalid_row_id());
    }

    int64_t external_id(int64_t phys) const {
        return static_cast<int64_t>(row_ids[static_cast<size_t>(phys)]);
    }

    void reserve_storage(int64_t new_capacity) {
        std::vector<float> new_x;
        if (store_float32) new_x.assign(static_cast<size_t>(new_capacity * d), 0.0f);
        std::vector<int8_t> new_codes_col(static_cast<size_t>(new_capacity * d), 0);
        std::vector<int8_t> new_codes_row;
        if (store_codes_row) new_codes_row.assign(static_cast<size_t>(new_capacity * d), 0);
        std::vector<float> new_scales(static_cast<size_t>(new_capacity), 1.0f);
        std::vector<float> new_norms(static_cast<size_t>(new_capacity), 0.0f);
        std::vector<float> new_residuals;
        if (store_residuals) new_residuals.assign(static_cast<size_t>(new_capacity), 0.0f);
        std::vector<uint8_t> new_alive(static_cast<size_t>(new_capacity), 0);
        std::vector<uint32_t> new_row_ids(static_cast<size_t>(new_capacity), invalid_row_id());

        for (int64_t i = 0; i < n; ++i) {
            if (store_float32) {
                std::copy(x.begin() + i * d, x.begin() + (i + 1) * d, new_x.begin() + i * d);
            }
            for (int64_t j = 0; j < d; ++j) {
                new_codes_col[static_cast<size_t>(j * new_capacity + i)] = codes_col[static_cast<size_t>(j * capacity + i)];
                if (store_codes_row) {
                    new_codes_row[static_cast<size_t>(i * d + j)] = codes_row[static_cast<size_t>(i * d + j)];
                }
            }
            new_scales[static_cast<size_t>(i)] = scales[static_cast<size_t>(i)];
            new_norms[static_cast<size_t>(i)] = norms[static_cast<size_t>(i)];
            if (store_residuals) new_residuals[static_cast<size_t>(i)] = residuals[static_cast<size_t>(i)];
            new_alive[static_cast<size_t>(i)] = alive[static_cast<size_t>(i)];
            new_row_ids[static_cast<size_t>(i)] = row_ids[static_cast<size_t>(i)];
        }

        x.swap(new_x);
        codes_col.swap(new_codes_col);
        codes_row.swap(new_codes_row);
        scales.swap(new_scales);
        norms.swap(new_norms);
        residuals.swap(new_residuals);
        alive.swap(new_alive);
        row_ids.swap(new_row_ids);
        capacity = new_capacity;
    }

    void ensure_capacity(int64_t needed) {
        if (needed <= capacity) return;
        int64_t new_capacity = std::max<int64_t>(needed, static_cast<int64_t>(std::ceil(static_cast<double>(capacity) * growth_factor)));
        if (new_capacity < 1) new_capacity = 1;
        reserve_storage(new_capacity);
    }

    int64_t append_row(const float *row, int64_t row_id) {
        if (!valid_row_id(row_id)) return -1;
        ensure_capacity(n + 1);
        const int64_t i = n;
        float max_abs = 0.0f;
        float norm2 = 0.0f;
        for (int64_t j = 0; j < d; ++j) {
            const float v = row[j];
            if (store_float32) x[static_cast<size_t>(i * d + j)] = v;
            max_abs = std::max(max_abs, std::abs(v));
            norm2 += v * v;
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        float residual2 = 0.0f;
        for (int64_t j = 0; j < d; ++j) {
            const float v = row[j];
            const int8_t code = quantize_value(v, scale);
            codes_col[static_cast<size_t>(j * capacity + i)] = code;
            if (store_codes_row) codes_row[static_cast<size_t>(i * d + j)] = code;
            if (store_residuals) {
                const float diff = v - static_cast<float>(code) * scale;
                residual2 += diff * diff;
            }
        }
        scales[static_cast<size_t>(i)] = scale;
        norms[static_cast<size_t>(i)] = std::sqrt(norm2);
        if (store_residuals) residuals[static_cast<size_t>(i)] = std::sqrt(residual2);
        alive[static_cast<size_t>(i)] = 1;
        const uint32_t stored_id = static_cast<uint32_t>(row_id);
        row_ids[static_cast<size_t>(i)] = stored_id;
        id_to_phys[stored_id] = i;
        ++n;
        ++live_count;
        next_row_id = std::max(next_row_id, row_id + 1);
        return row_id;
    }

    void compact() {
        const int64_t old_capacity = capacity;
        const int64_t new_capacity = std::max<int64_t>(1, live_count);
        std::vector<float> new_x;
        if (store_float32) new_x.assign(static_cast<size_t>(new_capacity * d), 0.0f);
        std::vector<int8_t> new_codes_col(static_cast<size_t>(new_capacity * d), 0);
        std::vector<int8_t> new_codes_row;
        if (store_codes_row) new_codes_row.assign(static_cast<size_t>(new_capacity * d), 0);
        std::vector<float> new_scales(static_cast<size_t>(new_capacity), 1.0f);
        std::vector<float> new_norms(static_cast<size_t>(new_capacity), 0.0f);
        std::vector<float> new_residuals;
        if (store_residuals) new_residuals.assign(static_cast<size_t>(new_capacity), 0.0f);
        std::vector<uint8_t> new_alive(static_cast<size_t>(new_capacity), 0);
        std::vector<uint32_t> new_row_ids(static_cast<size_t>(new_capacity), invalid_row_id());

        id_to_phys.clear();
        int64_t write = 0;
        for (int64_t i = 0; i < n; ++i) {
            if (!alive[static_cast<size_t>(i)]) continue;
            if (store_float32) {
                std::copy(x.begin() + i * d, x.begin() + (i + 1) * d, new_x.begin() + write * d);
            }
            for (int64_t j = 0; j < d; ++j) {
                new_codes_col[static_cast<size_t>(j * new_capacity + write)] = codes_col[static_cast<size_t>(j * old_capacity + i)];
                if (store_codes_row) {
                    new_codes_row[static_cast<size_t>(write * d + j)] = codes_row[static_cast<size_t>(i * d + j)];
                }
            }
            new_scales[static_cast<size_t>(write)] = scales[static_cast<size_t>(i)];
            new_norms[static_cast<size_t>(write)] = norms[static_cast<size_t>(i)];
            if (store_residuals) new_residuals[static_cast<size_t>(write)] = residuals[static_cast<size_t>(i)];
            new_alive[static_cast<size_t>(write)] = 1;
            const uint32_t id = row_ids[static_cast<size_t>(i)];
            new_row_ids[static_cast<size_t>(write)] = id;
            id_to_phys[id] = write;
            ++write;
        }
        x.swap(new_x);
        codes_col.swap(new_codes_col);
        codes_row.swap(new_codes_row);
        scales.swap(new_scales);
        norms.swap(new_norms);
        residuals.swap(new_residuals);
        alive.swap(new_alive);
        row_ids.swap(new_row_ids);
        capacity = new_capacity;
        n = write;
        live_count = write;
    }
};

struct SearchScratch {
    std::vector<float> block_scores;
    std::vector<int32_t> block_scores_i32;
    std::vector<float> scout_scores;
    std::vector<float> block_upper;
    std::vector<uint8_t> block_alive;
    std::vector<std::pair<float, int64_t>> heap;
    std::vector<std::pair<float, int64_t>> lower_heap;
    std::vector<std::pair<float, int64_t>> scout_heap;
    std::vector<std::pair<float, int64_t>> exact;
    std::vector<int64_t> rerank_ids;
    std::vector<int8_t> q_codes;
};

static thread_local SearchScratch scratch;

void *etar_storage_create(
    const float *x,
    int64_t n,
    int64_t d,
    double initial_capacity_factor,
    double growth_factor,
    int32_t store_residuals,
    int32_t store_float32,
    int32_t store_codes_row
) {
    if (!x || n < 0 || d <= 0) return nullptr;
    if (n >= static_cast<int64_t>(EtarStorageIndex::invalid_row_id())) return nullptr;
    auto idx = std::make_unique<EtarStorageIndex>();
    idx->n = 0;
    idx->d = d;
    idx->live_count = 0;
    idx->growth_factor = growth_factor > 1.0 ? growth_factor : 2.0;
    idx->store_residuals = store_residuals != 0;
    idx->store_float32 = store_float32 != 0;
    idx->store_codes_row = store_codes_row != 0;
    idx->capacity = std::max<int64_t>(1, static_cast<int64_t>(std::ceil(std::max<int64_t>(n, 1) * std::max(1.0, initial_capacity_factor))));
    if (idx->store_float32) idx->x.assign(static_cast<size_t>(idx->capacity * d), 0.0f);
    idx->codes_col.assign(static_cast<size_t>(idx->capacity * d), 0);
    if (idx->store_codes_row) idx->codes_row.assign(static_cast<size_t>(idx->capacity * d), 0);
    idx->scales.assign(static_cast<size_t>(idx->capacity), 1.0f);
    idx->norms.assign(static_cast<size_t>(idx->capacity), 0.0f);
    if (idx->store_residuals) idx->residuals.assign(static_cast<size_t>(idx->capacity), 0.0f);
    idx->alive.assign(static_cast<size_t>(idx->capacity), 0);
    idx->row_ids.assign(static_cast<size_t>(idx->capacity), EtarStorageIndex::invalid_row_id());
    for (int64_t i = 0; i < n; ++i) {
        idx->append_row(x + i * d, i);
    }
    return idx.release();
}

void etar_storage_free(void *handle) {
    delete static_cast<EtarStorageIndex *>(handle);
}

int64_t etar_storage_n(void *handle) {
    auto *idx = static_cast<EtarStorageIndex *>(handle);
    return idx ? idx->n : 0;
}

int64_t etar_storage_live_count(void *handle) {
    auto *idx = static_cast<EtarStorageIndex *>(handle);
    return idx ? idx->live_count : 0;
}

int64_t etar_storage_capacity(void *handle) {
    auto *idx = static_cast<EtarStorageIndex *>(handle);
    return idx ? idx->capacity : 0;
}

int64_t etar_storage_insert(void *handle, const float *row) {
    auto *idx = static_cast<EtarStorageIndex *>(handle);
    if (!idx || !row) return -1;
    const int64_t row_id = idx->next_row_id;
    return idx->append_row(row, row_id);
}

void etar_storage_delete(void *handle, int64_t row_id) {
    auto *idx = static_cast<EtarStorageIndex *>(handle);
    if (!idx) return;
    if (!EtarStorageIndex::valid_row_id(row_id)) return;
    const auto found = idx->id_to_phys.find(static_cast<uint32_t>(row_id));
    if (found == idx->id_to_phys.end()) return;
    const int64_t i = found->second;
    if (i >= 0 && i < idx->n && idx->alive[static_cast<size_t>(i)]) {
        idx->alive[static_cast<size_t>(i)] = 0;
        --idx->live_count;
    }
}

int64_t etar_storage_update(void *handle, int64_t row_id, const float *row) {
    etar_storage_delete(handle, row_id);
    return etar_storage_insert(handle, row);
}

void etar_storage_compact(void *handle) {
    auto *idx = static_cast<EtarStorageIndex *>(handle);
    if (idx) idx->compact();
}

EtarStorageStatus etar_storage_search_profile(
    void *handle,
    const float *q,
    const int64_t *head,
    int64_t h,
    int64_t k,
    int64_t r,
    int32_t use_tail,
    float alpha_tail,
    float tail_norm,
    int32_t no_rerank,
    int32_t prune_lower_bound,
    float prune_lower_scale,
    int32_t block_prune,
    int64_t block_prune_scout_h,
    float block_prune_beam_factor,
    float block_prune_gate,
    int64_t block_prune_block_size,
    int32_t sort_candidates,
    int32_t quantize_query,
    int32_t quantize_rerank_query,
    int64_t *out_ids,
    float *out_scores,
    int64_t *cand_ids,
    float *cand_scores,
    EtarStorageProfile *profile
) {
    auto *idx = static_cast<EtarStorageIndex *>(handle);
    EtarStorageStatus status{0, 0};
    if (profile) *profile = EtarStorageProfile{0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f};
    if (!idx || !q || !head || !out_ids || !out_scores || !cand_ids || !cand_scores || idx->live_count <= 0) return status;

    k = std::min<int64_t>(std::max<int64_t>(k, 0), idx->live_count);
    r = std::min<int64_t>(std::max<int64_t>(r, k), idx->live_count);
    h = std::min<int64_t>(std::max<int64_t>(h, 0), idx->d);
    const int64_t cand_n = r;
    const float tail_factor = use_tail ? alpha_tail * tail_norm : 0.0f;
    const bool use_global_lower_prune = prune_lower_bound == 1 && k > 0;
    const bool use_candidate_lower_prune = prune_lower_bound == 2 && k > 0;
    const bool use_any_lower_prune = use_global_lower_prune || use_candidate_lower_prune;
    const bool use_quantized_scan_query = quantize_query != 0;
    const bool use_quantized_rerank_query = quantize_rerank_query != 0;
    const bool need_quantized_query = use_quantized_scan_query ||
        (use_quantized_rerank_query && !no_rerank && !idx->store_float32);
    float q_scale = 1.0f;
    if (need_quantized_query) {
        auto &q_codes = scratch.q_codes;
        q_codes.resize(static_cast<size_t>(idx->d));
        const bool need_full_query_codes = use_quantized_rerank_query && !no_rerank && !idx->store_float32;
        float q_max_abs = 0.0f;
        for (int64_t j = 0; j < idx->d; ++j) {
            q_max_abs = std::max(q_max_abs, std::abs(q[j]));
        }
        q_scale = q_max_abs > 0.0f ? q_max_abs / 127.0f : 1.0f;
        if (need_full_query_codes) {
            for (int64_t j = 0; j < idx->d; ++j) {
                q_codes[static_cast<size_t>(j)] = quantize_value(q[j], q_scale);
            }
        } else {
            for (int64_t t = 0; t < h; ++t) {
                const int64_t j = head[t];
                q_codes[static_cast<size_t>(j)] = quantize_value(q[j], q_scale);
            }
        }
    }

    auto t0 = Clock::now();
    auto &heap = scratch.heap;
    auto &lower_heap = scratch.lower_heap;
    heap.clear();
    heap.reserve(static_cast<size_t>(cand_n));
    lower_heap.clear();
    if (use_any_lower_prune) lower_heap.reserve(static_cast<size_t>(k));

    const auto consider = [&](float s, int64_t i) {
        std::pair<float, int64_t> item{s, i};
        if (static_cast<int64_t>(heap.size()) < cand_n) {
            heap.emplace_back(item);
            std::push_heap(heap.begin(), heap.end(), worse_for_min_heap);
        } else if (cand_n > 0 && better_desc(item, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), worse_for_min_heap);
            heap.back() = item;
            std::push_heap(heap.begin(), heap.end(), worse_for_min_heap);
        }
    };
    const float lower_scale = std::max(0.0f, prune_lower_scale);
    const auto lower_score = [&](float partial_score, float bound) {
        return partial_score - lower_scale * bound;
    };
    const auto observe_lower = [&](float s, int64_t i) {
        if (static_cast<int64_t>(lower_heap.size()) < k) {
            lower_heap.emplace_back(s, i);
            std::push_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
        } else if (s > lower_heap.front().first || (s == lower_heap.front().first && i < lower_heap.front().second)) {
            std::pop_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
            lower_heap.back() = std::pair<float, int64_t>{s, i};
            std::push_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
        }
    };

    Clock::time_point t_scan;
    float block_prune_survival = 1.0f;
    float block_prune_used = 0.0f;
    float block_prune_fallback = 0.0f;

    if (block_prune == 1 && h > 1 && cand_n > 0) {
        const int64_t scout_h = std::max<int64_t>(1, std::min<int64_t>(block_prune_scout_h, h - 1));
        const int64_t prune_block_size = std::max<int64_t>(64, block_prune_block_size);
        const int64_t total_prune_blocks = (idx->n + prune_block_size - 1) / prune_block_size;
        const float beam_factor = std::max(1.0f, block_prune_beam_factor);
        const int64_t beam_n = std::min<int64_t>(
            idx->live_count,
            std::max<int64_t>(cand_n, static_cast<int64_t>(std::ceil(beam_factor * static_cast<float>(cand_n))))
        );
        float remaining_head_norm2 = 0.0f;
        for (int64_t t = scout_h; t < h; ++t) {
            const float qj = q[head[t]];
            remaining_head_norm2 += qj * qj;
        }
        const float scout_gamma = std::sqrt(remaining_head_norm2) + (use_tail ? tail_factor : 0.0f);

        auto &block_scores = scratch.block_scores;
        auto &scout_scores = scratch.scout_scores;
        auto &block_upper = scratch.block_upper;
        auto &block_alive = scratch.block_alive;
        auto &scout_heap = scratch.scout_heap;
        scout_scores.assign(static_cast<size_t>(idx->n), 0.0f);
        block_upper.assign(static_cast<size_t>(total_prune_blocks), -std::numeric_limits<float>::infinity());
        block_alive.assign(static_cast<size_t>(total_prune_blocks), 0);
        scout_heap.clear();
        scout_heap.reserve(static_cast<size_t>(beam_n));

        const auto observe_scout = [&](float s, int64_t i) {
            std::pair<float, int64_t> item{s, i};
            if (static_cast<int64_t>(scout_heap.size()) < beam_n) {
                scout_heap.emplace_back(item);
                std::push_heap(scout_heap.begin(), scout_heap.end(), worse_for_min_heap);
            } else if (beam_n > 0 && better_desc(item, scout_heap.front())) {
                std::pop_heap(scout_heap.begin(), scout_heap.end(), worse_for_min_heap);
                scout_heap.back() = item;
                std::push_heap(scout_heap.begin(), scout_heap.end(), worse_for_min_heap);
            }
        };

        const int64_t scan_block_size = 4096;
        if (static_cast<int64_t>(block_scores.size()) < scan_block_size) {
            block_scores.resize(static_cast<size_t>(scan_block_size));
        }
        for (int64_t base = 0; base < idx->n; base += scan_block_size) {
            const int64_t len = std::min<int64_t>(scan_block_size, idx->n - base);
            std::fill(block_scores.begin(), block_scores.begin() + len, 0.0f);
            for (int64_t t = 0; t < scout_h; ++t) {
                const int64_t j = head[t];
                const int8_t *col = idx->codes_col.data() + j * idx->capacity + base;
                const float qj = q[j];
                if (qj == 0.0f) continue;
                #pragma omp simd
                for (int64_t off = 0; off < len; ++off) {
                    block_scores[static_cast<size_t>(off)] += qj * static_cast<float>(col[off]);
                }
            }
            for (int64_t off = 0; off < len; ++off) {
                const int64_t i = base + off;
                if (!idx->alive[static_cast<size_t>(i)]) continue;
                const float partial_score = idx->scales[static_cast<size_t>(i)] * block_scores[static_cast<size_t>(off)];
                scout_scores[static_cast<size_t>(i)] = partial_score;
                const float upper = partial_score + scout_gamma * idx->norms[static_cast<size_t>(i)];
                observe_scout(upper, i);
                const int64_t b = i / prune_block_size;
                if (upper > block_upper[static_cast<size_t>(b)]) {
                    block_upper[static_cast<size_t>(b)] = upper;
                }
            }
        }

        if (static_cast<int64_t>(scout_heap.size()) >= std::min<int64_t>(beam_n, cand_n)) {
            const float theta = scout_heap.front().first;
            int64_t alive_blocks = 0;
            for (int64_t b = 0; b < total_prune_blocks; ++b) {
                if (block_upper[static_cast<size_t>(b)] >= theta) {
                    block_alive[static_cast<size_t>(b)] = 1;
                    ++alive_blocks;
                }
            }
            block_prune_survival = total_prune_blocks > 0
                ? static_cast<float>(alive_blocks) / static_cast<float>(total_prune_blocks)
                : 1.0f;
            const float gate = std::max(0.0f, std::min(1.0f, block_prune_gate));
            if (alive_blocks > 0 && block_prune_survival <= gate) {
                block_prune_used = 1.0f;
                if (static_cast<int64_t>(block_scores.size()) < prune_block_size) {
                    block_scores.resize(static_cast<size_t>(prune_block_size));
                }
                for (int64_t b = 0; b < total_prune_blocks; ++b) {
                    if (!block_alive[static_cast<size_t>(b)]) continue;
                    const int64_t base = b * prune_block_size;
                    const int64_t len = std::min<int64_t>(prune_block_size, idx->n - base);
                    std::fill(block_scores.begin(), block_scores.begin() + len, 0.0f);
                    for (int64_t t = scout_h; t < h; ++t) {
                        const int64_t j = head[t];
                        const int8_t *col = idx->codes_col.data() + j * idx->capacity + base;
                        const float qj = q[j];
                        if (qj == 0.0f) continue;
                        #pragma omp simd
                        for (int64_t off = 0; off < len; ++off) {
                            block_scores[static_cast<size_t>(off)] += qj * static_cast<float>(col[off]);
                        }
                    }
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        if (!idx->alive[static_cast<size_t>(i)]) continue;
                        const float partial_score = scout_scores[static_cast<size_t>(i)] +
                            idx->scales[static_cast<size_t>(i)] * block_scores[static_cast<size_t>(off)];
                        const float bound = use_tail ? tail_factor * idx->norms[static_cast<size_t>(i)] : 0.0f;
                        if (use_global_lower_prune) observe_lower(lower_score(partial_score, bound), i);
                        consider(partial_score + bound, i);
                    }
                }
                t_scan = Clock::now();
                goto finalize_candidates;
            }
            block_prune_fallback = 1.0f;
        } else {
            block_prune_survival = 1.0f;
            block_prune_fallback = 1.0f;
        }
    }

    {
        constexpr int64_t scan_block_size = 4096;
        const bool all_alive = idx->live_count == idx->n;
        if (use_quantized_scan_query) {
            auto &block_scores_i32 = scratch.block_scores_i32;
            if (static_cast<int64_t>(block_scores_i32.size()) < scan_block_size) {
                block_scores_i32.resize(static_cast<size_t>(scan_block_size));
            }
            const int8_t *q_codes = scratch.q_codes.data();
            for (int64_t base = 0; base < idx->n; base += scan_block_size) {
                const int64_t len = std::min<int64_t>(scan_block_size, idx->n - base);
                std::fill(block_scores_i32.begin(), block_scores_i32.begin() + len, 0);
                for (int64_t t = 0; t < h; ++t) {
                    const int64_t j = head[t];
                    const int8_t *col = idx->codes_col.data() + j * idx->capacity + base;
                    const int32_t qj = static_cast<int32_t>(q_codes[j]);
                    add_i8_column_scaled_i32(block_scores_i32.data(), col, len, qj);
                }

                if (all_alive && !use_tail) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        const float s = idx->scales[static_cast<size_t>(i)] * q_scale *
                            static_cast<float>(block_scores_i32[static_cast<size_t>(off)]);
                        if (use_global_lower_prune) observe_lower(s, i);
                        consider(s, i);
                    }
                } else if (all_alive && use_tail && !use_global_lower_prune) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        const float s = idx->scales[static_cast<size_t>(i)] * q_scale *
                            static_cast<float>(block_scores_i32[static_cast<size_t>(off)]) +
                            tail_factor * idx->norms[static_cast<size_t>(i)];
                        consider(s, i);
                    }
                } else {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        if (!all_alive && !idx->alive[static_cast<size_t>(i)]) continue;
                        const float partial_score = idx->scales[static_cast<size_t>(i)] * q_scale *
                            static_cast<float>(block_scores_i32[static_cast<size_t>(off)]);
                        const float bound = use_tail ? tail_factor * idx->norms[static_cast<size_t>(i)] : 0.0f;
                        if (use_global_lower_prune) observe_lower(lower_score(partial_score, bound), i);
                        consider(partial_score + bound, i);
                    }
                }
            }
        } else {
            auto &block_scores = scratch.block_scores;
            if (static_cast<int64_t>(block_scores.size()) < scan_block_size) {
                block_scores.resize(static_cast<size_t>(scan_block_size));
            }
            for (int64_t base = 0; base < idx->n; base += scan_block_size) {
                const int64_t len = std::min<int64_t>(scan_block_size, idx->n - base);
                std::fill(block_scores.begin(), block_scores.begin() + len, 0.0f);
                for (int64_t t = 0; t < h; ++t) {
                    const int64_t j = head[t];
                    const int8_t *col = idx->codes_col.data() + j * idx->capacity + base;
                    const float qj = q[j];
                    if (qj == 0.0f) continue;
                    #pragma omp simd
                    for (int64_t off = 0; off < len; ++off) {
                        block_scores[static_cast<size_t>(off)] += qj * static_cast<float>(col[off]);
                    }
                }

                if (all_alive && !use_tail) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        const float s = idx->scales[static_cast<size_t>(i)] * block_scores[static_cast<size_t>(off)];
                        if (use_global_lower_prune) observe_lower(s, i);
                        consider(s, i);
                    }
                } else if (all_alive && use_tail && !use_global_lower_prune) {
                    #pragma omp simd
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        block_scores[static_cast<size_t>(off)] =
                            idx->scales[static_cast<size_t>(i)] * block_scores[static_cast<size_t>(off)] +
                            tail_factor * idx->norms[static_cast<size_t>(i)];
                    }
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                } else {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        if (!all_alive && !idx->alive[static_cast<size_t>(i)]) continue;
                        const float partial_score =
                            idx->scales[static_cast<size_t>(i)] * block_scores[static_cast<size_t>(off)];
                        const float bound = use_tail ? tail_factor * idx->norms[static_cast<size_t>(i)] : 0.0f;
                        if (use_global_lower_prune) observe_lower(lower_score(partial_score, bound), i);
                        consider(partial_score + bound, i);
                    }
                }
            }
        }
    }
    t_scan = Clock::now();

finalize_candidates:
    int64_t got_cands = static_cast<int64_t>(heap.size());
    if (use_candidate_lower_prune && !heap.empty()) {
        lower_heap.clear();
        lower_heap.reserve(static_cast<size_t>(k));
        for (const auto &item : heap) {
            const int64_t phys = item.second;
            const float bound = use_tail ? tail_factor * idx->norms[static_cast<size_t>(phys)] : 0.0f;
            const float lower = item.first - (1.0f + lower_scale) * bound;
            if (static_cast<int64_t>(lower_heap.size()) < k) {
                lower_heap.emplace_back(lower, phys);
                std::push_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
            } else if (lower > lower_heap.front().first || (lower == lower_heap.front().first && phys < lower_heap.front().second)) {
                std::pop_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
                lower_heap.back() = std::pair<float, int64_t>{lower, phys};
                std::push_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
            }
        }
    }
    if (use_any_lower_prune && !lower_heap.empty()) {
        const float threshold = lower_heap.front().first;
        auto end_it = std::remove_if(heap.begin(), heap.end(), [&](const std::pair<float, int64_t> &item) {
            return item.first < threshold;
        });
        heap.erase(end_it, heap.end());
        got_cands = static_cast<int64_t>(heap.size());
    }
    std::sort(heap.begin(), heap.end(), better_desc);
    status.candidates = static_cast<int32_t>(got_cands);
    for (int32_t i = 0; i < status.candidates; ++i) {
        const int64_t phys = heap[static_cast<size_t>(i)].second;
        cand_ids[i] = idx->external_id(phys);
        cand_scores[i] = heap[static_cast<size_t>(i)].first;
    }
    if (sort_candidates == 0) {
        // The heap is already sorted for deterministic output. The flag is kept for API parity.
    }
    auto t_top = Clock::now();

    if (no_rerank) {
        const int64_t take = std::min<int64_t>(k, heap.size());
        for (int64_t i = 0; i < take; ++i) {
            const int64_t phys = heap[static_cast<size_t>(i)].second;
            out_ids[i] = idx->external_id(phys);
            out_scores[i] = heap[static_cast<size_t>(i)].first;
        }
        status.results = static_cast<int32_t>(take);
    } else {
        auto &exact = scratch.exact;
        exact.clear();
        exact.reserve(heap.size());
        const auto rerank_score = [&](int64_t phys) {
            if (idx->store_float32) {
                return exact_dot(idx->x.data(), q, phys, idx->d);
            }
            if (idx->store_codes_row) {
                return use_quantized_rerank_query
                    ? quantized_dot_row_q8(idx->codes_row, idx->scales, scratch.q_codes.data(), q_scale, phys, idx->d)
                    : quantized_dot_row(idx->codes_row, idx->scales, q, phys, idx->d);
            }
            return use_quantized_rerank_query
                ? quantized_dot_col_q8(idx->codes_col, idx->scales, scratch.q_codes.data(), q_scale, phys, idx->d, idx->capacity)
                : quantized_dot_col(idx->codes_col, idx->scales, q, phys, idx->d, idx->capacity);
        };
        if (sort_candidates) {
            auto &rerank_ids = scratch.rerank_ids;
            rerank_ids.resize(static_cast<size_t>(got_cands));
            for (int64_t i = 0; i < got_cands; ++i) rerank_ids[static_cast<size_t>(i)] = heap[static_cast<size_t>(i)].second;
            std::sort(rerank_ids.begin(), rerank_ids.end());
            for (int64_t i = 0; i < got_cands; ++i) {
                const int64_t phys = rerank_ids[static_cast<size_t>(i)];
                exact.push_back({rerank_score(phys), phys});
            }
        } else {
            for (int64_t i = 0; i < got_cands; ++i) {
                const int64_t phys = heap[static_cast<size_t>(i)].second;
                exact.push_back({rerank_score(phys), phys});
            }
        }
        const int64_t take = std::min<int64_t>(k, exact.size());
        if (take > 0) {
            std::partial_sort(exact.begin(), exact.begin() + take, exact.end(), better_desc);
        }
        for (int64_t i = 0; i < take; ++i) {
            const int64_t phys = exact[static_cast<size_t>(i)].second;
            out_ids[i] = idx->external_id(phys);
            out_scores[i] = exact[static_cast<size_t>(i)].first;
        }
        status.results = static_cast<int32_t>(take);
    }
    auto t_rerank = Clock::now();

    if (profile) {
        profile->scan_ms = elapsed_ms(t0, t_scan);
        profile->top_r_ms = elapsed_ms(t_scan, t_top);
        profile->rerank_ms = elapsed_ms(t_top, t_rerank);
        profile->total_native_ms = elapsed_ms(t0, t_rerank);
        profile->block_prune_survival = block_prune_survival;
        profile->block_prune_used = block_prune_used;
        profile->block_prune_fallback = block_prune_fallback;
    }
    return status;
}

}  // extern "C"
