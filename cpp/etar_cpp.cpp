#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <immintrin.h>
#include <limits>
#include <utility>
#include <vector>

extern "C" {

struct EtarStatus {
    int32_t candidates;
    int32_t results;
};

struct EtarProfile {
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

static float exact_dot(const float *x, const float *q, int64_t row, int64_t d) {
    const float *xr = x + row * d;
    float s = 0.0f;
    for (int64_t j = 0; j < d; ++j) s += xr[j] * q[j];
    return s;
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

struct Scratch {
    std::vector<float> block_scores;
    std::vector<int32_t> block_scores_i32;
    std::vector<float> scout_scores;
    std::vector<float> block_upper;
    std::vector<uint8_t> block_alive;
    std::vector<float> q_work;
	    std::vector<int32_t> q_work_i32;
	    std::vector<std::pair<float, int64_t>> heap;
	    std::vector<std::pair<float, int64_t>> lower_heap;
	    std::vector<std::pair<float, int64_t>> scout_heap;
	    std::vector<std::pair<float, int64_t>> exact;
	    std::vector<int64_t> rerank_ids;
	};

static thread_local Scratch scratch;

static bool better_desc(const std::pair<float, int64_t> &a, const std::pair<float, int64_t> &b) {
    if (a.first == b.first) return a.second < b.second;
    return a.first > b.first;
}

static bool worse_for_min_heap(const std::pair<float, int64_t> &a, const std::pair<float, int64_t> &b) {
    if (a.first == b.first) return a.second > b.second;
    return a.first > b.first;
}

static EtarStatus etar_search_impl(
    const float *x,
    const float *x_scan,
    const int8_t *c_col,
    const float *scales,
    const float *scan_scales,
    const float *norms,
    const float *residuals,
    const uint8_t *alive,
    const float *q,
    const int64_t *head,
    int64_t h,
    int64_t n,
    int64_t d,
    int64_t scan_stride,
    int64_t k,
    int64_t r,
    int32_t use_tail,
    float alpha_tail,
    float tail_norm,
	    float alpha_q,
	    int32_t query_mode,
	    int32_t scan_mode,
	    int32_t no_rerank,
	    int32_t prune_lower_bound,
	    float prune_lower_scale,
	    int32_t block_prune,
	    int64_t block_prune_scout_h,
	    float block_prune_beam_factor,
	    float block_prune_gate,
	    int64_t block_prune_block_size,
	    int32_t sort_rerank_candidates,
	    int64_t *out_ids,
	    float *out_scores,
	    int64_t *cand_ids,
    float *cand_scores,
    EtarProfile *profile
) {
    const auto t0 = Clock::now();
    const float q_head_norm = [&]() {
        float s = 0.0f;
        for (int64_t t = 0; t < h; ++t) s += q[head[t]] * q[head[t]];
        return std::sqrt(s);
    }();
    const float qrisk_factor = alpha_q * q_head_norm * std::sqrt(static_cast<float>(h) / std::max<int64_t>(d, 1));

    auto &block_scores = scratch.block_scores;
    auto &block_scores_i32 = scratch.block_scores_i32;
    auto &q_work = scratch.q_work;
    auto &q_work_i32 = scratch.q_work_i32;
    q_work.resize(static_cast<size_t>(h));
    q_work_i32.resize(static_cast<size_t>(h));
    float query_i8_scale = 1.0f;
    const bool use_i32_scan = query_mode == 1 && scan_mode == 0;

    if (query_mode == 1) {
        float max_abs = 0.0f;
        for (int64_t t = 0; t < h; ++t) max_abs = std::max(max_abs, std::fabs(q[head[t]]));
        const float tq = max_abs > 0.0f ? max_abs / 127.0f : 1.0f / 127.0f;
        query_i8_scale = tq;
        for (int64_t t = 0; t < h; ++t) {
            const int64_t j = head[t];
            int32_t b = static_cast<int32_t>(std::lrint(q[j] / tq));
            b = std::max(-127, std::min(127, b));
            q_work_i32[static_cast<size_t>(t)] = b;
            q_work[static_cast<size_t>(t)] = tq * static_cast<float>(b);
        }
    } else {
        for (int64_t t = 0; t < h; ++t) q_work[static_cast<size_t>(t)] = q[head[t]];
    }

    const int64_t cand_n = std::min<int64_t>(r, n);
    const bool use_global_lower_prune = prune_lower_bound == 1 && k > 0;
    const bool use_candidate_lower_prune = prune_lower_bound == 2 && k > 0;
    const bool use_any_lower_prune = use_global_lower_prune || use_candidate_lower_prune;
    auto &heap = scratch.heap;
    auto &lower_heap = scratch.lower_heap;
    heap.clear();
    heap.reserve(static_cast<size_t>(cand_n));
    lower_heap.clear();
    if (use_any_lower_prune) {
        lower_heap.reserve(static_cast<size_t>(k));
    }

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
    const auto lower_score = [&](float partial, float bound) {
        return partial - lower_scale * bound;
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

    constexpr int64_t block_size = 4096;
    if (static_cast<int64_t>(block_scores.size()) < block_size) {
        block_scores.resize(static_cast<size_t>(block_size));
    }
    if (static_cast<int64_t>(block_scores_i32.size()) < block_size) {
        block_scores_i32.resize(static_cast<size_t>(block_size));
    }
    const float tail_factor = alpha_tail * tail_norm;
    Clock::time_point t_scan;
    float block_prune_survival = 1.0f;
    float block_prune_used = 0.0f;
    float block_prune_fallback = 0.0f;

    if (block_prune == 1 && query_mode == 0 && scan_mode == 0 && h > 1 && cand_n > 0) {
        const int64_t scout_h = std::max<int64_t>(1, std::min<int64_t>(block_prune_scout_h, h - 1));
        const int64_t prune_block_size = std::max<int64_t>(64, block_prune_block_size);
        const int64_t total_prune_blocks = (n + prune_block_size - 1) / prune_block_size;
        const float beam_factor = std::max(1.0f, block_prune_beam_factor);
        const int64_t beam_n = std::min<int64_t>(n, std::max<int64_t>(cand_n, static_cast<int64_t>(std::ceil(beam_factor * static_cast<float>(cand_n)))));
        float remaining_head_norm2 = 0.0f;
        for (int64_t t = scout_h; t < h; ++t) {
            const float qj = q[head[t]];
            remaining_head_norm2 += qj * qj;
        }
        const float scout_gamma = std::sqrt(remaining_head_norm2) + (use_tail ? tail_factor : 0.0f);

        auto &scout_scores = scratch.scout_scores;
        auto &block_upper = scratch.block_upper;
        auto &block_alive = scratch.block_alive;
        auto &scout_heap = scratch.scout_heap;
        scout_scores.assign(static_cast<size_t>(n), 0.0f);
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
        for (int64_t base = 0; base < n; base += scan_block_size) {
            const int64_t len = std::min<int64_t>(scan_block_size, n - base);
            std::fill(block_scores.begin(), block_scores.begin() + len, 0.0f);
            for (int64_t t = 0; t < scout_h; ++t) {
                const int64_t j = head[t];
                const int8_t *col = c_col + j * scan_stride + base;
                const float qj = q_work[static_cast<size_t>(t)];
                if (qj == 0.0f) continue;
                #pragma omp simd
                for (int64_t off = 0; off < len; ++off) {
                    block_scores[static_cast<size_t>(off)] += qj * static_cast<float>(col[off]);
                }
            }
            for (int64_t off = 0; off < len; ++off) {
                const int64_t i = base + off;
                if (alive && alive[i] == 0) continue;
                const float partial = scales[i] * block_scores[static_cast<size_t>(off)];
                scout_scores[static_cast<size_t>(i)] = partial;
                float upper = partial + scout_gamma * norms[i];
                if (alpha_q != 0.0f && residuals) upper += qrisk_factor * residuals[i];
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
                    const int64_t len = std::min<int64_t>(prune_block_size, n - base);
                    std::fill(block_scores.begin(), block_scores.begin() + len, 0.0f);
                    for (int64_t t = scout_h; t < h; ++t) {
                        const int64_t j = head[t];
                        const int8_t *col = c_col + j * scan_stride + base;
                        const float qj = q_work[static_cast<size_t>(t)];
                        if (qj == 0.0f) continue;
                        #pragma omp simd
                        for (int64_t off = 0; off < len; ++off) {
                            block_scores[static_cast<size_t>(off)] += qj * static_cast<float>(col[off]);
                        }
                    }
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        if (alive && alive[i] == 0) continue;
                        const float partial = scout_scores[static_cast<size_t>(i)] + scales[i] * block_scores[static_cast<size_t>(off)];
                        float bound = 0.0f;
                        if (use_tail) bound += tail_factor * norms[i];
                        if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[i];
                        if (use_global_lower_prune) observe_lower(lower_score(partial, bound), i);
                        consider(partial + bound, i);
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

    for (int64_t base = 0; base < n; base += block_size) {
        const int64_t len = std::min<int64_t>(block_size, n - base);
        if (scan_mode == 4 || scan_mode == 5) {
            const bool column_quantized = scan_mode == 5;
            for (int64_t off = 0; off < len; ++off) {
                const int64_t i = base + off;
                if (alive && alive[i] == 0) continue;
                const int8_t *row_codes = c_col + i * d;
                float partial = 0.0f;
                for (int64_t t = 0; t < h; ++t) {
                    const int64_t j = head[t];
                    float qj = q_work[static_cast<size_t>(t)];
                    if (column_quantized) qj *= scan_scales[j];
                    partial += qj * static_cast<float>(row_codes[j]);
                }
                if (!column_quantized) partial *= scales[i];
                float bound = 0.0f;
                if (use_tail) bound += tail_factor * norms[i];
                if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[i];
                if (use_global_lower_prune) observe_lower(lower_score(partial, bound), i);
                consider(partial + bound, i);
            }
        } else if (scan_mode == 2 || scan_mode == 3) {
            std::fill(block_scores.begin(), block_scores.begin() + len, 0.0f);
            if (scan_mode == 3) {
                for (int64_t t = 0; t < h; ++t) {
                    const int64_t j = head[t];
                    const float *col = x_scan + j * scan_stride + base;
                    const float qj = q_work[static_cast<size_t>(t)];
                    if (qj == 0.0f) continue;
                    #pragma omp simd
                    for (int64_t off = 0; off < len; ++off) {
                        block_scores[static_cast<size_t>(off)] += qj * col[off];
                    }
                }
            } else {
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t row = base + off;
                    const float *xr = x + row * d;
                    float acc = 0.0f;
                    for (int64_t t = 0; t < h; ++t) {
                        acc += q_work[static_cast<size_t>(t)] * xr[head[t]];
                    }
                    block_scores[static_cast<size_t>(off)] = acc;
                }
            }
            if (!alive && !use_tail && alpha_q == 0.0f) {
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    const float s = block_scores[static_cast<size_t>(off)];
                    if (use_global_lower_prune) observe_lower(s, i);
                    consider(s, i);
                }
            } else {
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    if (alive && alive[i] == 0) continue;
                    const float partial = block_scores[static_cast<size_t>(off)];
                    float bound = 0.0f;
                    if (use_tail) bound += tail_factor * norms[i];
                    if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[i];
                    if (use_global_lower_prune) observe_lower(lower_score(partial, bound), i);
                    consider(partial + bound, i);
                }
            }
        } else if (use_i32_scan) {
            std::fill(block_scores_i32.begin(), block_scores_i32.begin() + len, 0);

                for (int64_t t = 0; t < h; ++t) {
                    const int64_t j = head[t];
                    const int8_t *col = c_col + j * scan_stride + base;
                    const int32_t qj = q_work_i32[static_cast<size_t>(t)];
                    add_i8_column_scaled_i32(block_scores_i32.data(), col, len, qj);
                }

            if (!alive && !use_tail && alpha_q == 0.0f) {
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    const float s = query_i8_scale * scales[i] * static_cast<float>(block_scores_i32[static_cast<size_t>(off)]);
                    if (use_global_lower_prune) observe_lower(s, i);
                    consider(s, i);
                }
            } else if (!alive && use_tail && alpha_q == 0.0f) {
                #pragma omp simd
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    block_scores[static_cast<size_t>(off)] =
                        query_i8_scale * scales[i] * static_cast<float>(block_scores_i32[static_cast<size_t>(off)]) +
                        tail_factor * norms[i];
                }
                if (use_global_lower_prune) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        const float partial = query_i8_scale * scales[i] * static_cast<float>(block_scores_i32[static_cast<size_t>(off)]);
                        const float bound = tail_factor * norms[i];
                        observe_lower(lower_score(partial, bound), i);
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                } else {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                }
            } else if (!alive) {
                #pragma omp simd
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    float s = query_i8_scale * scales[i] * static_cast<float>(block_scores_i32[static_cast<size_t>(off)]);
                    if (use_tail) s += tail_factor * norms[i];
                    if (alpha_q != 0.0f && residuals) s += qrisk_factor * residuals[i];
                    block_scores[static_cast<size_t>(off)] = s;
                }
                if (use_global_lower_prune) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        const float partial = query_i8_scale * scales[i] * static_cast<float>(block_scores_i32[static_cast<size_t>(off)]);
                        float bound = 0.0f;
                        if (use_tail) bound += tail_factor * norms[i];
                        if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[i];
                        observe_lower(lower_score(partial, bound), i);
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                } else {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                }
            } else {
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    if (alive && alive[i] == 0) continue;
                    float partial = query_i8_scale * scales[i] * static_cast<float>(block_scores_i32[static_cast<size_t>(off)]);
                    float bound = 0.0f;
                    if (use_tail) bound += tail_factor * norms[i];
                    if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[i];
                    if (use_global_lower_prune) observe_lower(lower_score(partial, bound), i);
                    consider(partial + bound, i);
                }
            }
        } else {
            std::fill(block_scores.begin(), block_scores.begin() + len, 0.0f);

            for (int64_t t = 0; t < h; ++t) {
                const int64_t j = head[t];
                const int8_t *col = c_col + j * scan_stride + base;
                const float qj = q_work[static_cast<size_t>(t)] * (scan_mode == 1 ? scan_scales[j] : 1.0f);
                if (qj == 0.0f) continue;
                #pragma omp simd
                for (int64_t off = 0; off < len; ++off) {
                    block_scores[static_cast<size_t>(off)] += qj * static_cast<float>(col[off]);
                }
            }

            if (!alive && !use_tail && alpha_q == 0.0f) {
                if (scan_mode == 1) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        const float s = block_scores[static_cast<size_t>(off)];
                        if (use_global_lower_prune) observe_lower(s, i);
                        consider(s, i);
                    }
                } else {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        const float s = scales[i] * block_scores[static_cast<size_t>(off)];
                        if (use_global_lower_prune) observe_lower(s, i);
                        consider(s, i);
                    }
                }
            } else if (!alive && use_tail && alpha_q == 0.0f) {
                #pragma omp simd
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    float s = block_scores[static_cast<size_t>(off)];
                    if (scan_mode != 1) s *= scales[i];
                    block_scores[static_cast<size_t>(off)] = s + tail_factor * norms[i];
                }
                if (use_global_lower_prune) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        float partial = block_scores[static_cast<size_t>(off)] - tail_factor * norms[i];
                        const float bound = tail_factor * norms[i];
                        observe_lower(lower_score(partial, bound), i);
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                } else {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                }
            } else if (!alive) {
                #pragma omp simd
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    float s = block_scores[static_cast<size_t>(off)];
                    if (scan_mode != 1) s *= scales[i];
                    if (use_tail) s += tail_factor * norms[i];
                    if (alpha_q != 0.0f && residuals) s += qrisk_factor * residuals[i];
                    block_scores[static_cast<size_t>(off)] = s;
                }
                if (use_global_lower_prune) {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        float bound = 0.0f;
                        if (use_tail) bound += tail_factor * norms[i];
                        if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[i];
                        const float partial = block_scores[static_cast<size_t>(off)] - bound;
                        observe_lower(lower_score(partial, bound), i);
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                } else {
                    for (int64_t off = 0; off < len; ++off) {
                        const int64_t i = base + off;
                        consider(block_scores[static_cast<size_t>(off)], i);
                    }
                }
            } else {
                for (int64_t off = 0; off < len; ++off) {
                    const int64_t i = base + off;
                    if (alive && alive[i] == 0) continue;
                    float partial = block_scores[static_cast<size_t>(off)];
                    if (scan_mode != 1) partial *= scales[i];
                    float bound = 0.0f;
                    if (use_tail) bound += tail_factor * norms[i];
                    if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[i];
                    if (use_global_lower_prune) observe_lower(lower_score(partial, bound), i);
                    consider(partial + bound, i);
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
	            const int64_t row = item.second;
	            float bound = 0.0f;
	            if (use_tail) bound += tail_factor * norms[row];
	            if (alpha_q != 0.0f && residuals) bound += qrisk_factor * residuals[row];
	            const float lower = item.first - (1.0f + lower_scale) * bound;
	            if (static_cast<int64_t>(lower_heap.size()) < k) {
	                lower_heap.emplace_back(lower, row);
	                std::push_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
	            } else if (lower > lower_heap.front().first || (lower == lower_heap.front().first && row < lower_heap.front().second)) {
	                std::pop_heap(lower_heap.begin(), lower_heap.end(), worse_for_min_heap);
	                lower_heap.back() = std::pair<float, int64_t>{lower, row};
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
	    if (cand_n > 0) {
	        std::sort(heap.begin(), heap.end(), better_desc);
	    }
	    const auto t_topr = Clock::now();
	
	    for (int64_t i = 0; i < got_cands; ++i) {
	        cand_scores[i] = heap[static_cast<size_t>(i)].first;
	        cand_ids[i] = heap[static_cast<size_t>(i)].second;
	    }
	    for (int64_t i = got_cands; i < r; ++i) {
	        cand_scores[i] = -std::numeric_limits<float>::infinity();
	        cand_ids[i] = -1;
	    }

	    if (no_rerank) {
	        const int64_t out_n = std::min<int64_t>(k, got_cands);
	        for (int64_t i = 0; i < out_n; ++i) {
	            out_scores[i] = cand_scores[i];
	            out_ids[i] = cand_ids[i];
	        }
	        for (int64_t i = out_n; i < k; ++i) {
	            out_scores[i] = -std::numeric_limits<float>::infinity();
	            out_ids[i] = -1;
	        }
	        const auto t_done = Clock::now();
	        if (profile) {
	            profile->scan_ms = elapsed_ms(t0, t_scan);
	            profile->top_r_ms = elapsed_ms(t_scan, t_topr);
	            profile->rerank_ms = 0.0f;
	            profile->total_native_ms = elapsed_ms(t0, t_done);
	            profile->block_prune_survival = block_prune_survival;
	            profile->block_prune_used = block_prune_used;
	            profile->block_prune_fallback = block_prune_fallback;
	        }
	        return EtarStatus{static_cast<int32_t>(got_cands), static_cast<int32_t>(out_n)};
	    }

	    auto &exact = scratch.exact;
	    exact.clear();
	    exact.reserve(static_cast<size_t>(got_cands));
	    if (sort_rerank_candidates) {
	        auto &rerank_ids = scratch.rerank_ids;
	        rerank_ids.resize(static_cast<size_t>(got_cands));
	        for (int64_t i = 0; i < got_cands; ++i) rerank_ids[static_cast<size_t>(i)] = cand_ids[i];
	        std::sort(rerank_ids.begin(), rerank_ids.end());
	        for (int64_t i = 0; i < got_cands; ++i) {
	            const int64_t row = rerank_ids[static_cast<size_t>(i)];
	            exact.emplace_back(exact_dot(x, q, row, d), row);
	        }
	    } else {
	        for (int64_t i = 0; i < got_cands; ++i) {
	            exact.emplace_back(exact_dot(x, q, cand_ids[i], d), cand_ids[i]);
	        }
	    }

    const int64_t out_n = std::min<int64_t>(k, static_cast<int64_t>(exact.size()));
    if (out_n > 0) {
        std::partial_sort(exact.begin(), exact.begin() + out_n, exact.end(), better_desc);
    }
    for (int64_t i = 0; i < out_n; ++i) {
        out_scores[i] = exact[static_cast<size_t>(i)].first;
        out_ids[i] = exact[static_cast<size_t>(i)].second;
    }
    for (int64_t i = out_n; i < k; ++i) {
        out_scores[i] = -std::numeric_limits<float>::infinity();
        out_ids[i] = -1;
    }
    const auto t_done = Clock::now();

    if (profile) {
        profile->scan_ms = elapsed_ms(t0, t_scan);
        profile->top_r_ms = elapsed_ms(t_scan, t_topr);
        profile->rerank_ms = elapsed_ms(t_topr, t_done);
        profile->total_native_ms = elapsed_ms(t0, t_done);
        profile->block_prune_survival = block_prune_survival;
        profile->block_prune_used = block_prune_used;
        profile->block_prune_fallback = block_prune_fallback;
    }

    return EtarStatus{static_cast<int32_t>(got_cands), static_cast<int32_t>(out_n)};
}

EtarStatus etar_search(
    const float *x,
    const float *x_scan,
    const int8_t *c_col,
    const float *scales,
    const float *scan_scales,
    const float *norms,
    const float *residuals,
    const uint8_t *alive,
    const float *q,
    const int64_t *head,
    int64_t h,
    int64_t n,
    int64_t d,
    int64_t scan_stride,
    int64_t k,
    int64_t r,
    int32_t use_tail,
    float alpha_tail,
    float tail_norm,
	    float alpha_q,
	    int32_t query_mode,
	    int32_t scan_mode,
	    int32_t no_rerank,
	    int32_t prune_lower_bound,
	    float prune_lower_scale,
	    int32_t block_prune,
	    int64_t block_prune_scout_h,
	    float block_prune_beam_factor,
	    float block_prune_gate,
	    int64_t block_prune_block_size,
	    int32_t sort_rerank_candidates,
	    int64_t *out_ids,
	    float *out_scores,
    int64_t *cand_ids,
    float *cand_scores
	) {
	    return etar_search_impl(x, x_scan, c_col, scales, scan_scales, norms, residuals, alive, q, head, h, n, d, scan_stride, k, r,
	                            use_tail, alpha_tail, tail_norm, alpha_q, query_mode, scan_mode, no_rerank,
	                            prune_lower_bound, prune_lower_scale, block_prune, block_prune_scout_h,
	                            block_prune_beam_factor, block_prune_gate, block_prune_block_size,
	                            sort_rerank_candidates, out_ids, out_scores, cand_ids, cand_scores, nullptr);
	}

EtarStatus etar_search_profile(
    const float *x,
    const float *x_scan,
    const int8_t *c_col,
    const float *scales,
    const float *scan_scales,
    const float *norms,
    const float *residuals,
    const uint8_t *alive,
    const float *q,
    const int64_t *head,
    int64_t h,
    int64_t n,
    int64_t d,
    int64_t scan_stride,
    int64_t k,
    int64_t r,
    int32_t use_tail,
    float alpha_tail,
    float tail_norm,
	    float alpha_q,
	    int32_t query_mode,
	    int32_t scan_mode,
	    int32_t no_rerank,
	    int32_t prune_lower_bound,
	    float prune_lower_scale,
	    int32_t block_prune,
	    int64_t block_prune_scout_h,
	    float block_prune_beam_factor,
	    float block_prune_gate,
	    int64_t block_prune_block_size,
	    int32_t sort_rerank_candidates,
	    int64_t *out_ids,
	    float *out_scores,
    int64_t *cand_ids,
    float *cand_scores,
    EtarProfile *profile
	) {
	    return etar_search_impl(x, x_scan, c_col, scales, scan_scales, norms, residuals, alive, q, head, h, n, d, scan_stride, k, r,
	                            use_tail, alpha_tail, tail_norm, alpha_q, query_mode, scan_mode, no_rerank,
	                            prune_lower_bound, prune_lower_scale, block_prune, block_prune_scout_h,
	                            block_prune_beam_factor, block_prune_gate, block_prune_block_size,
	                            sort_rerank_candidates, out_ids, out_scores, cand_ids, cand_scores, profile);
	}
	
	void etar_search_batch(
    const float *x,
    const float *x_scan,
    const int8_t *c_col,
    const float *scales,
    const float *scan_scales,
    const float *norms,
    const float *residuals,
    const uint8_t *alive,
    const float *queries,
    const int64_t *heads,
    const int64_t *head_offsets,
    int64_t qn,
    int64_t n,
    int64_t d,
    int64_t scan_stride,
    int64_t k,
    int64_t r,
    const int32_t *use_tail,
    const float *alpha_tail,
    const float *tail_norm,
	    const float *alpha_q,
	    const int32_t *query_mode,
	    const int32_t *no_rerank,
	    const int32_t *prune_lower_bound,
	    const float *prune_lower_scale,
	    int32_t block_prune,
	    int64_t block_prune_scout_h,
	    float block_prune_beam_factor,
	    float block_prune_gate,
	    int64_t block_prune_block_size,
	    int32_t sort_rerank_candidates,
	    int64_t *out_ids,
	    float *out_scores,
    int64_t *cand_ids,
    float *cand_scores,
    int32_t scan_mode,
    EtarProfile *profiles
) {
    for (int64_t qi = 0; qi < qn; ++qi) {
        const int64_t start = head_offsets[qi];
        const int64_t end = head_offsets[qi + 1];
	        etar_search_impl(
	            x, x_scan, c_col, scales, scan_scales, norms, residuals, alive, queries + qi * d, heads + start, end - start,
	            n, d, scan_stride, k, r, use_tail[qi], alpha_tail[qi], tail_norm[qi], alpha_q[qi], query_mode[qi], scan_mode,
	            no_rerank[qi], prune_lower_bound[qi], prune_lower_scale[qi], block_prune, block_prune_scout_h,
	            block_prune_beam_factor, block_prune_gate, block_prune_block_size, sort_rerank_candidates,
	            out_ids + qi * k, out_scores + qi * k, cand_ids + qi * r, cand_scores + qi * r,
	            profiles ? profiles + qi : nullptr
	        );
    }
}

}
