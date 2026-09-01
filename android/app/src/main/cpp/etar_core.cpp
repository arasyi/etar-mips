#include "etar_core.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__aarch64__) || defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace etar {
namespace {

using Clock = std::chrono::steady_clock;

enum class Method { Exact, FullI8, Etar, EtarLm };

struct Pair {
    float score;
    int32_t id;
};

struct SearchScratch {
    std::vector<int32_t> order;
    std::vector<int32_t> accum;
    std::vector<float> scan;
    std::vector<Pair> candidates;
    std::vector<int32_t> output;
};

bool better(const Pair& a, const Pair& b) {
    return a.score == b.score ? a.id < b.id : a.score > b.score;
}

struct MethodResult {
    std::string name;
    double build_ms = 0.0;
    uint64_t representation_bytes = 0;
    uint64_t workspace_bytes = 0;
    double recall = 1.0;
    int64_t measured_queries = 0;
    double total_latency_ms = 0.0;
    double measurement_wall_ms = 0.0;
    std::string energy_status = "unsupported";
    double energy_joules = 0.0;
    std::vector<double> latency_ms;
    std::vector<int32_t> first_ids;
};

struct IdleEnergyResult {
    std::string status = "not_requested";
    double duration_ms = 0.0;
    double energy_joules = 0.0;
};

double elapsed_ms(Clock::time_point a, Clock::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

std::string json_escape(const std::string& value) {
    std::ostringstream out;
    for (unsigned char c : value) {
        switch (c) {
            case '\\': out << "\\\\"; break;
            case '"': out << "\\\""; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (c < 0x20) {
                    out << "\\u" << std::hex << std::setw(4)
                        << std::setfill('0') << static_cast<int>(c) << std::dec;
                } else {
                    out << c;
                }
        }
    }
    return out.str();
}

void validate(BenchmarkConfig& c) {
    if (c.corpus_size <= 0 || c.corpus_size > 2000000) {
        throw std::invalid_argument("corpus size must be between 1 and 2,000,000");
    }
    if (c.dimension <= 0 || c.dimension > 4096) {
        throw std::invalid_argument("dimension must be between 1 and 4,096");
    }
    const uint64_t elements = static_cast<uint64_t>(c.corpus_size) * c.dimension;
    if (elements > 400000000ULL) {
        throw std::invalid_argument("corpus exceeds the 400 million element safety limit");
    }
    if (c.query_count <= 0 || c.query_count > 10000) {
        throw std::invalid_argument("query count must be between 1 and 10,000");
    }
    if (c.warmup_queries < 0 || c.warmup_queries > 10000) {
        throw std::invalid_argument("warm-up query count is invalid");
    }
    if (c.repetitions <= 0 || c.repetitions > 1000) {
        throw std::invalid_argument("repetitions must be between 1 and 1,000");
    }
    if (c.k <= 0) throw std::invalid_argument("K must be positive");
    c.k = static_cast<int32_t>(std::min<int64_t>(c.k, c.corpus_size));
    c.rerank = std::max(c.k, c.rerank);
    c.rerank = static_cast<int32_t>(std::min<int64_t>(c.rerank, c.corpus_size));
    if (!(c.rho > 0.0f && c.rho <= 1.0f)) {
        throw std::invalid_argument("rho must be in (0, 1]");
    }
    c.head_cap = std::max(1, std::min(c.head_cap, c.dimension));
    if (c.alpha_min < 0.0f || c.alpha_max < c.alpha_min) {
        throw std::invalid_argument("tail alpha bounds are invalid");
    }
    if (c.lambda_tail < 0.0f) {
        throw std::invalid_argument("tail lambda must not be negative");
    }
    if (c.measurement_seconds < 0.0f || c.measurement_seconds > 3600.0f) {
        throw std::invalid_argument("measurement duration must be between 0 and 3,600 seconds");
    }
    if (c.cpu_core < -1 || c.cpu_core >= 1024) {
        throw std::invalid_argument("CPU core must be -1 (unpinned) or a non-negative core index");
    }
    if (c.distribution != "dense_gaussian" && c.distribution != "sparse10" &&
        c.distribution != "mixed_dropout_heavytail" &&
        c.distribution != "norm_heavy") {
        throw std::invalid_argument("unknown synthetic distribution");
    }
}

std::vector<Method> parse_order(const std::string& text) {
    std::vector<Method> out;
    size_t start = 0;
    while (start <= text.size()) {
        const size_t comma = text.find(',', start);
        const std::string token = text.substr(start, comma - start);
        if (token == "exact") out.push_back(Method::Exact);
        else if (token == "full_i8") out.push_back(Method::FullI8);
        else if (token == "etar") out.push_back(Method::Etar);
        else if (token == "etar_lm") out.push_back(Method::EtarLm);
        else throw std::invalid_argument(
            "method order must contain exact, full_i8, etar, and etar_lm");
        if (comma == std::string::npos) break;
        start = comma + 1;
    }
    std::vector<Method> sorted = out;
    std::sort(sorted.begin(), sorted.end(), [](Method a, Method b) {
        return static_cast<int>(a) < static_cast<int>(b);
    });
    if (sorted != std::vector<Method>{Method::Exact, Method::FullI8,
                                      Method::Etar, Method::EtarLm}) {
        throw std::invalid_argument("method order must contain each method exactly once");
    }
    return out;
}

std::vector<float> generate_matrix(int64_t rows, int32_t d, int64_t seed,
                                   const std::string& distribution,
                                   bool queries) {
    std::mt19937_64 rng(static_cast<uint64_t>(seed) ^
                        (queries ? 0x9e3779b97f4a7c15ULL : 0ULL));
    std::normal_distribution<float> normal(0.0f, 1.0f);
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    std::student_t_distribution<float> student_t(3.0f);
    std::lognormal_distribution<float> lognormal(0.0f, 1.25f);
    std::vector<float> data(static_cast<size_t>(rows) * d);

    for (float& v : data) {
        v = distribution == "mixed_dropout_heavytail" ? student_t(rng) : normal(rng);
    }
    if (distribution == "sparse10") {
        for (float& v : data) {
            if (uniform(rng) >= 0.10f) v = 0.0f;
        }
    } else if (distribution == "mixed_dropout_heavytail") {
        for (float& v : data) {
            if (uniform(rng) <= 0.35f) v = 0.0f;
        }
    } else if (distribution == "norm_heavy" && !queries) {
        for (int64_t i = 0; i < rows; ++i) {
            const float row_scale = lognormal(rng);
            for (int32_t j = 0; j < d; ++j) {
                data[static_cast<size_t>(i) * d + j] *= row_scale;
            }
        }
    }
    return data;
}

void add_column_float(float* scores, const int8_t* column, int64_t n, float qj) {
    int64_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
    const float32x4_t qv = vdupq_n_f32(qj);
    for (; i + 7 < n; i += 8) {
        const int8x8_t c8 = vld1_s8(column + i);
        const int16x8_t c16 = vmovl_s8(c8);
        const int32x4_t lo32 = vmovl_s16(vget_low_s16(c16));
        const int32x4_t hi32 = vmovl_s16(vget_high_s16(c16));
        float32x4_t lo = vld1q_f32(scores + i);
        float32x4_t hi = vld1q_f32(scores + i + 4);
        lo = vmlaq_f32(lo, vcvtq_f32_s32(lo32), qv);
        hi = vmlaq_f32(hi, vcvtq_f32_s32(hi32), qv);
        vst1q_f32(scores + i, lo);
        vst1q_f32(scores + i + 4, hi);
    }
#endif
    for (; i < n; ++i) scores[i] += static_cast<float>(column[i]) * qj;
}

void add_column_i32(int32_t* scores, const int8_t* column, int64_t n, int32_t qj) {
    int64_t i = 0;
#if defined(__aarch64__) || defined(__ARM_NEON)
    const int16x8_t q16 = vdupq_n_s16(static_cast<int16_t>(qj));
    for (; i + 7 < n; i += 8) {
        const int16x8_t c16 = vmovl_s8(vld1_s8(column + i));
        const int16x8_t product = vmulq_s16(c16, q16);
        int32x4_t lo = vld1q_s32(scores + i);
        int32x4_t hi = vld1q_s32(scores + i + 4);
        lo = vaddq_s32(lo, vmovl_s16(vget_low_s16(product)));
        hi = vaddq_s32(hi, vmovl_s16(vget_high_s16(product)));
        vst1q_s32(scores + i, lo);
        vst1q_s32(scores + i + 4, hi);
    }
#endif
    for (; i < n; ++i) scores[i] += static_cast<int32_t>(column[i]) * qj;
}

class Index {
public:
    Index(Method method, std::vector<float>&& x, int64_t n, int32_t d)
        : method_(method), n_(n), d_(d) {
        if (method == Method::Exact) {
            x_ = std::move(x);
            representation_bytes_ = static_cast<uint64_t>(n_) * d_ * sizeof(float);
            return;
        }

        const bool needs_tail_statistics = method != Method::FullI8;
        codes_col_.resize(static_cast<size_t>(n_) * d_);
        if (method == Method::EtarLm) codes_row_.resize(static_cast<size_t>(n_) * d_);
        scales_.resize(static_cast<size_t>(n_));
        if (needs_tail_statistics) {
            norms_.resize(static_cast<size_t>(n_));
            omega_.assign(static_cast<size_t>(d_), 0.0f);
        }

        for (int64_t i = 0; i < n_; ++i) {
            const float* row = x.data() + i * d_;
            float max_abs = 0.0f;
            double norm2 = 0.0;
            for (int32_t j = 0; j < d_; ++j) {
                max_abs = std::max(max_abs, std::fabs(row[j]));
                if (needs_tail_statistics) {
                    norm2 += static_cast<double>(row[j]) * row[j];
                }
            }
            const float norm = needs_tail_statistics
                ? static_cast<float>(std::sqrt(norm2)) : 0.0f;
            const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
            scales_[static_cast<size_t>(i)] = scale;
            if (needs_tail_statistics) norms_[static_cast<size_t>(i)] = norm;
            for (int32_t j = 0; j < d_; ++j) {
                int code = static_cast<int>(std::lrint(row[j] / scale));
                code = std::max(-127, std::min(127, code));
                codes_col_[static_cast<size_t>(j) * n_ + i] = static_cast<int8_t>(code);
                if (!codes_row_.empty()) {
                    codes_row_[static_cast<size_t>(i) * d_ + j] = static_cast<int8_t>(code);
                }
                if (norm > 0.0f) {
                    omega_[static_cast<size_t>(j)] +=
                        static_cast<float>((static_cast<double>(row[j]) * row[j]) / norm2);
                }
            }
        }
        for (float& value : omega_) value /= static_cast<float>(n_);
        if (method == Method::Etar) x_ = std::move(x);
        else {
            x.clear();
            x.shrink_to_fit();
        }

        representation_bytes_ = static_cast<uint64_t>(codes_col_.size() + codes_row_.size());
        representation_bytes_ += static_cast<uint64_t>(scales_.size() + norms_.size() + omega_.size()) * sizeof(float);
        representation_bytes_ += static_cast<uint64_t>(n_) * (sizeof(uint8_t) + sizeof(uint32_t));
        representation_bytes_ += static_cast<uint64_t>(x_.size()) * sizeof(float);
    }

    uint64_t representation_bytes() const { return representation_bytes_; }

    uint64_t workspace_bytes(int32_t k, int32_t rerank, bool quantize_query) const {
        if (method_ == Method::Exact) {
            return static_cast<uint64_t>(k) * sizeof(Pair);
        }
        if (method_ == Method::FullI8) {
            return static_cast<uint64_t>(n_) * sizeof(int32_t) +
                   static_cast<uint64_t>(k) * sizeof(Pair);
        }
        return static_cast<uint64_t>(n_) * sizeof(float) +
               (quantize_query ? static_cast<uint64_t>(n_) * sizeof(int32_t) : 0ULL) +
               static_cast<uint64_t>(rerank) * sizeof(Pair) +
               static_cast<uint64_t>(d_) * sizeof(int32_t);
    }

    const std::vector<int32_t>& search(const float* q, const BenchmarkConfig& c,
                                       SearchScratch& scratch) const {
        if (method_ == Method::Exact) return exact(q, c.k, scratch);
        if (method_ == Method::FullI8) return full_i8(q, c.k, scratch);

        std::vector<int32_t>& order = scratch.order;
        order.resize(static_cast<size_t>(d_));
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [q](int32_t a, int32_t b) {
            return q[a] * q[a] > q[b] * q[b];
        });
        double total = 0.0;
        for (int32_t j = 0; j < d_; ++j) total += static_cast<double>(q[j]) * q[j];
        double retained2 = 0.0;
        int32_t h = 0;
        while (h < d_ && h < c.head_cap) {
            const int32_t j = order[static_cast<size_t>(h++)];
            retained2 += static_cast<double>(q[j]) * q[j];
            if (total <= 0.0 || retained2 / total >= c.rho) break;
        }
        h = std::max(1, h);
        const float tail_norm = static_cast<float>(std::sqrt(std::max(0.0, total - retained2)));
        double sigma2 = 0.0;
        for (int32_t t = h; t < d_; ++t) {
            const int32_t j = order[static_cast<size_t>(t)];
            sigma2 += static_cast<double>(q[j]) * q[j] * omega_[static_cast<size_t>(j)];
        }
        const double ratio = std::max(1.0, static_cast<double>(n_) / c.rerank);
        const float z = static_cast<float>(std::sqrt(2.0 * std::log(ratio)));
        float alpha = c.lambda_tail * z * static_cast<float>(std::sqrt(sigma2)) /
                      (tail_norm + 1e-12f);
        alpha = std::min(c.alpha_max, std::max(c.alpha_min, alpha));

        std::vector<float>& scan = scratch.scan;
        scan.assign(static_cast<size_t>(n_), 0.0f);
        if (c.quantize_query) {
            float qmax = 0.0f;
            for (int32_t t = 0; t < h; ++t) qmax = std::max(qmax, std::fabs(q[order[t]]));
            const float qscale = qmax > 0.0f ? qmax / 127.0f : 1.0f;
            std::vector<int32_t>& accum = scratch.accum;
            accum.assign(static_cast<size_t>(n_), 0);
            for (int32_t t = 0; t < h; ++t) {
                const int32_t j = order[static_cast<size_t>(t)];
                int qcode = static_cast<int>(std::lrint(q[j] / qscale));
                qcode = std::max(-127, std::min(127, qcode));
                add_column_i32(accum.data(), codes_col_.data() + static_cast<size_t>(j) * n_, n_, qcode);
            }
            for (int64_t i = 0; i < n_; ++i) {
                scan[static_cast<size_t>(i)] = scales_[static_cast<size_t>(i)] * qscale *
                    static_cast<float>(accum[static_cast<size_t>(i)]);
            }
        } else {
            for (int32_t t = 0; t < h; ++t) {
                const int32_t j = order[static_cast<size_t>(t)];
                add_column_float(scan.data(), codes_col_.data() + static_cast<size_t>(j) * n_, n_, q[j]);
            }
            for (int64_t i = 0; i < n_; ++i) {
                scan[static_cast<size_t>(i)] *= scales_[static_cast<size_t>(i)];
            }
        }
        for (int64_t i = 0; i < n_; ++i) {
            scan[static_cast<size_t>(i)] += alpha * tail_norm * norms_[static_cast<size_t>(i)];
        }

        std::vector<Pair>& candidates = scratch.candidates;
        candidates.clear();
        candidates.reserve(static_cast<size_t>(c.rerank));
        for (int64_t i = 0; i < n_; ++i) {
            const Pair item{scan[static_cast<size_t>(i)], static_cast<int32_t>(i)};
            if (static_cast<int32_t>(candidates.size()) < c.rerank) {
                candidates.push_back(item);
                std::push_heap(candidates.begin(), candidates.end(), better);
            } else if (better(item, candidates.front())) {
                std::pop_heap(candidates.begin(), candidates.end(), better);
                candidates.back() = item;
                std::push_heap(candidates.begin(), candidates.end(), better);
            }
        }
        const int32_t r = std::min<int64_t>(c.rerank, n_);

        for (Pair& candidate : candidates) {
            candidate.score = method_ == Method::Etar
                ? dot_float(q, candidate.id)
                : dot_quantized(q, candidate.id);
        }
        const int32_t take = std::min(c.k, r);
        std::partial_sort(candidates.begin(), candidates.begin() + take, candidates.end(), better);
        scratch.output.resize(static_cast<size_t>(take));
        for (int32_t i = 0; i < take; ++i) scratch.output[static_cast<size_t>(i)] = candidates[static_cast<size_t>(i)].id;
        return scratch.output;
    }

private:
    const std::vector<int32_t>& full_i8(const float* q, int32_t k,
                                        SearchScratch& scratch) const {
        float qmax = 0.0f;
        for (int32_t j = 0; j < d_; ++j) qmax = std::max(qmax, std::fabs(q[j]));
        const float qscale = qmax > 0.0f ? qmax / 127.0f : 1.0f;
        std::vector<int32_t>& accum = scratch.accum;
        accum.assign(static_cast<size_t>(n_), 0);
        for (int32_t j = 0; j < d_; ++j) {
            int qcode = static_cast<int>(std::lrint(q[j] / qscale));
            qcode = std::max(-127, std::min(127, qcode));
            add_column_i32(accum.data(),
                           codes_col_.data() + static_cast<size_t>(j) * n_,
                           n_, qcode);
        }

        std::vector<Pair>& scores = scratch.candidates;
        const int32_t take = std::min<int64_t>(k, n_);
        scores.clear();
        scores.reserve(static_cast<size_t>(take));
        for (int64_t i = 0; i < n_; ++i) {
            const Pair item{scales_[static_cast<size_t>(i)] * qscale *
                                static_cast<float>(accum[static_cast<size_t>(i)]),
                            static_cast<int32_t>(i)};
            if (static_cast<int32_t>(scores.size()) < take) {
                scores.push_back(item);
                std::push_heap(scores.begin(), scores.end(), better);
            } else if (better(item, scores.front())) {
                std::pop_heap(scores.begin(), scores.end(), better);
                scores.back() = item;
                std::push_heap(scores.begin(), scores.end(), better);
            }
        }
        std::sort(scores.begin(), scores.end(), better);
        scratch.output.resize(static_cast<size_t>(take));
        for (int32_t i = 0; i < take; ++i) {
            scratch.output[static_cast<size_t>(i)] = scores[static_cast<size_t>(i)].id;
        }
        return scratch.output;
    }

    float dot_float(const float* q, int32_t row) const {
        const float* x = x_.data() + static_cast<size_t>(row) * d_;
        float score = 0.0f;
        for (int32_t j = 0; j < d_; ++j) score += x[j] * q[j];
        return score;
    }

    float dot_quantized(const float* q, int32_t row) const {
        const int8_t* x = codes_row_.data() + static_cast<size_t>(row) * d_;
        float score = 0.0f;
        for (int32_t j = 0; j < d_; ++j) score += static_cast<float>(x[j]) * q[j];
        return scales_[static_cast<size_t>(row)] * score;
    }

    const std::vector<int32_t>& exact(const float* q, int32_t k,
                                      SearchScratch& scratch) const {
        std::vector<Pair>& scores = scratch.candidates;
        const int32_t take = std::min<int64_t>(k, n_);
        scores.clear();
        scores.reserve(static_cast<size_t>(take));
        for (int64_t i = 0; i < n_; ++i) {
            const Pair item{dot_float(q, i), static_cast<int32_t>(i)};
            if (static_cast<int32_t>(scores.size()) < take) {
                scores.push_back(item);
                std::push_heap(scores.begin(), scores.end(), better);
            } else if (better(item, scores.front())) {
                std::pop_heap(scores.begin(), scores.end(), better);
                scores.back() = item;
                std::push_heap(scores.begin(), scores.end(), better);
            }
        }
        std::sort(scores.begin(), scores.end(), better);
        scratch.output.resize(static_cast<size_t>(take));
        for (int32_t i = 0; i < take; ++i) scratch.output[static_cast<size_t>(i)] = scores[static_cast<size_t>(i)].id;
        return scratch.output;
    }

    Method method_;
    int64_t n_;
    int32_t d_;
    uint64_t representation_bytes_ = 0;
    std::vector<float> x_;
    std::vector<int8_t> codes_col_;
    std::vector<int8_t> codes_row_;
    std::vector<float> scales_;
    std::vector<float> norms_;
    std::vector<float> omega_;
};

std::string method_name(Method method) {
    if (method == Method::Exact) return "exact_f32";
    if (method == Method::FullI8) return "full_i8_scan";
    if (method == Method::Etar) return "etar";
    return "etar_lm";
}

MethodResult run_method(Method method, const BenchmarkConfig& c,
                        const std::vector<float>& queries,
                        const TraceHooks& trace) {
    MethodResult result;
    result.name = method_name(method);
    std::vector<float> corpus = generate_matrix(c.corpus_size, c.dimension, c.seed,
                                                c.distribution, false);
    const auto build_start = Clock::now();
    Index index(method, std::move(corpus), c.corpus_size, c.dimension);
    const auto build_end = Clock::now();
    result.build_ms = elapsed_ms(build_start, build_end);
    result.representation_bytes = index.representation_bytes();
    result.workspace_bytes = index.workspace_bytes(c.k, c.rerank, c.quantize_query);
    SearchScratch scratch;

    for (int32_t i = 0; i < c.warmup_queries; ++i) {
        const int32_t qi = i % c.query_count;
        index.search(queries.data() + static_cast<size_t>(qi) * c.dimension, c, scratch);
    }

    const std::string trace_name = "ETAR_BENCH_" + result.name;
    const double energy_start = trace.energy_joules
        ? trace.energy_joules() : std::numeric_limits<double>::quiet_NaN();
    if (trace.begin) trace.begin(trace_name.c_str());
    const auto measurement_start = Clock::now();
    const double minimum_ms = static_cast<double>(c.measurement_seconds) * 1000.0;
    int64_t iteration = 0;
    do {
        const int32_t qi = static_cast<int32_t>(iteration % c.query_count);
        const auto start = Clock::now();
        const std::vector<int32_t>& ids = index.search(
            queries.data() + static_cast<size_t>(qi) * c.dimension, c, scratch);
        const auto end = Clock::now();
        const double latency = elapsed_ms(start, end);
        result.total_latency_ms += latency;
        // Bound memory during long power runs. Once full, this becomes a
        // rolling window whose percentiles reflect the thermally relevant end
        // of the run; the mean below still covers every measured query.
        constexpr size_t max_latency_samples = 1000000;
        if (result.latency_ms.size() < max_latency_samples) {
            result.latency_ms.push_back(latency);
        } else {
            result.latency_ms[static_cast<size_t>(iteration) % max_latency_samples] = latency;
        }
        if (iteration < c.query_count) {
            result.first_ids.insert(result.first_ids.end(), ids.begin(), ids.end());
        }
        ++iteration;
    } while (iteration < static_cast<int64_t>(c.query_count) * c.repetitions ||
             elapsed_ms(measurement_start, Clock::now()) < minimum_ms);
    result.measurement_wall_ms = elapsed_ms(measurement_start, Clock::now());
    if (trace.end) trace.end();
    const double energy_end = trace.energy_joules
        ? trace.energy_joules() : std::numeric_limits<double>::quiet_NaN();
    if (std::isfinite(energy_start) && std::isfinite(energy_end)) {
        const double consumed = energy_start - energy_end;
        if (consumed > 0.0) {
            result.energy_status = "measured";
            result.energy_joules = consumed;
        } else {
            result.energy_status = "counter_not_decreasing";
        }
    }
    result.measured_queries = iteration;
    return result;
}

double percentile(std::vector<double> values, double p) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double position = p * static_cast<double>(values.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(position));
    const size_t hi = static_cast<size_t>(std::ceil(position));
    const double fraction = position - lo;
    return values[lo] * (1.0 - fraction) + values[hi] * fraction;
}

double recall_at_k(const std::vector<int32_t>& truth,
                   const std::vector<int32_t>& approximate,
                   int32_t queries, int32_t k) {
    if (truth.size() < static_cast<size_t>(queries) * k ||
        approximate.size() < static_cast<size_t>(queries) * k) return 0.0;
    int64_t hits = 0;
    for (int32_t qi = 0; qi < queries; ++qi) {
        for (int32_t a = 0; a < k; ++a) {
            const int32_t id = approximate[static_cast<size_t>(qi) * k + a];
            for (int32_t t = 0; t < k; ++t) {
                if (truth[static_cast<size_t>(qi) * k + t] == id) {
                    ++hits;
                    break;
                }
            }
        }
    }
    return static_cast<double>(hits) / (static_cast<double>(queries) * k);
}

IdleEnergyResult measure_idle_energy(const BenchmarkConfig& c,
                                     const std::vector<MethodResult>& results,
                                     const TraceHooks& trace) {
    IdleEnergyResult idle;
    if (c.measurement_seconds <= 0.0f) return idle;
    if (!trace.energy_joules) {
        idle.status = "unsupported";
        return idle;
    }
    for (const MethodResult& result : results) {
        idle.duration_ms = std::max(idle.duration_ms, result.measurement_wall_ms);
    }
    if (idle.duration_ms <= 0.0) return idle;

    const double energy_start = trace.energy_joules();
    if (trace.begin) trace.begin("ETAR_BENCH_idle");
    const auto idle_start = Clock::now();
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(idle.duration_ms));
    idle.duration_ms = elapsed_ms(idle_start, Clock::now());
    if (trace.end) trace.end();
    const double energy_end = trace.energy_joules();
    if (std::isfinite(energy_start) && std::isfinite(energy_end)) {
        const double consumed = energy_start - energy_end;
        if (consumed > 0.0) {
            idle.status = "measured";
            idle.energy_joules = consumed;
        } else {
            idle.status = "counter_not_decreasing";
        }
    } else {
        idle.status = "unsupported";
    }
    return idle;
}

std::string to_json(const BenchmarkConfig& c, const std::vector<MethodResult>& results,
                    const IdleEnergyResult& idle) {
    std::ostringstream out;
    out << std::fixed << std::setprecision(6);
    out << "{\"schema_version\":2,\"native_build\":{";
#if defined(__aarch64__)
    out << "\"architecture\":\"aarch64\",\"scan_kernel\":\"neon\"";
#elif defined(__x86_64__)
    out << "\"architecture\":\"x86_64\",\"scan_kernel\":\"scalar\"";
#else
    out << "\"architecture\":\"other\",\"scan_kernel\":\"scalar\"";
#endif
    out << ",\"compiler\":\"" << json_escape(__VERSION__) << "\""
        << ",\"compile_flags\":\"-O3 -ffast-math -fno-math-errno\""
        << ",\"threads\":1},\"config\":{";
    out << "\"corpus_size\":" << c.corpus_size
        << ",\"dimension\":" << c.dimension
        << ",\"query_count\":" << c.query_count
        << ",\"warmup_queries\":" << c.warmup_queries
        << ",\"repetitions\":" << c.repetitions
        << ",\"k\":" << c.k
        << ",\"rerank\":" << c.rerank
        << ",\"rho\":" << c.rho
        << ",\"head_cap\":" << c.head_cap
        << ",\"lambda_tail\":" << c.lambda_tail
        << ",\"alpha_min\":" << c.alpha_min
        << ",\"alpha_max\":" << c.alpha_max
        << ",\"seed\":" << c.seed
        << ",\"measurement_seconds\":" << c.measurement_seconds
        << ",\"quantize_scan_query\":" << (c.quantize_query ? "true" : "false")
        << ",\"cpu_core\":" << c.cpu_core
        << ",\"distribution\":\"" << json_escape(c.distribution) << "\""
        << ",\"dataset_definition_source\":\"src/etar/data.py\""
        << ",\"method_order\":\"" << json_escape(c.method_order) << "\"},";
    out << "\"methods\":[";
    for (size_t i = 0; i < results.size(); ++i) {
        const MethodResult& r = results[i];
        if (i) out << ',';
        const double mean = r.measured_queries == 0 ? 0.0 :
            r.total_latency_ms / static_cast<double>(r.measured_queries);
        const double duration_seconds = r.measurement_wall_ms / 1000.0;
        const bool energy_measured = r.energy_status == "measured";
        const bool idle_measured = idle.status == "measured" && idle.duration_ms > 0.0;
        const double idle_power = idle_measured
            ? idle.energy_joules / (idle.duration_ms / 1000.0) : 0.0;
        const double adjusted_energy = energy_measured && idle_measured
            ? std::max(0.0, r.energy_joules - idle_power * duration_seconds) : 0.0;
        out << "{\"name\":\"" << r.name << "\""
            << ",\"build_ms\":" << r.build_ms
            << ",\"representation_bytes\":" << r.representation_bytes
            << ",\"workspace_bytes\":" << r.workspace_bytes
            << ",\"measured_queries\":" << r.measured_queries
            << ",\"measurement_wall_ms\":" << r.measurement_wall_ms
            << ",\"throughput_qps\":" << (r.measurement_wall_ms > 0.0 ?
                1000.0 * r.measured_queries / r.measurement_wall_ms : 0.0)
            << ",\"recall_at_k\":" << r.recall
            << ",\"energy_counter_status\":\"" << r.energy_status << "\""
            << ",\"energy_joules\":";
        if (energy_measured) out << r.energy_joules;
        else out << "null";
        out << ",\"energy_joules_per_query\":";
        if (energy_measured && r.measured_queries > 0) {
            out << r.energy_joules / static_cast<double>(r.measured_queries);
        } else {
            out << "null";
        }
        out << ",\"average_power_watts\":";
        if (energy_measured && duration_seconds > 0.0) {
            out << r.energy_joules / duration_seconds;
        } else {
            out << "null";
        }
        out << ",\"idle_adjusted_energy_joules\":";
        if (energy_measured && idle_measured) out << adjusted_energy;
        else out << "null";
        out << ",\"idle_adjusted_energy_joules_per_query\":";
        if (energy_measured && idle_measured && r.measured_queries > 0) {
            out << adjusted_energy / static_cast<double>(r.measured_queries);
        } else {
            out << "null";
        }
        out << ",\"idle_adjusted_average_power_watts\":";
        if (energy_measured && idle_measured && duration_seconds > 0.0) {
            out << adjusted_energy / duration_seconds;
        } else {
            out << "null";
        }
        out << ",\"latency_ms\":{"
            << "\"mean\":" << mean
            << ",\"median\":" << percentile(r.latency_ms, 0.50)
            << ",\"p95\":" << percentile(r.latency_ms, 0.95)
            << ",\"p99\":" << percentile(r.latency_ms, 0.99)
            << ",\"min\":" << percentile(r.latency_ms, 0.00)
            << ",\"max\":" << percentile(r.latency_ms, 1.00) << "}}";
    }
    out << "],\"idle_energy_baseline\":{";
    out << "\"status\":\"" << idle.status << "\""
        << ",\"duration_ms\":" << idle.duration_ms
        << ",\"energy_joules\":";
    if (idle.status == "measured") out << idle.energy_joules;
    else out << "null";
    out << ",\"average_power_watts\":";
    if (idle.status == "measured" && idle.duration_ms > 0.0) {
        out << idle.energy_joules / (idle.duration_ms / 1000.0);
    } else {
        out << "null";
    }
    out << "},\"notes\":["
        << "\"Representation bytes exclude temporary query/workspace allocations.\","
        << "\"Energy is measured externally over ETAR_BENCH_* Android trace sections.\","
        << "\"In-app joules use Android's remaining-energy counter and may be unavailable.\","
        << "\"Synthetic data generation and warm-up are excluded from query latency.\""
        << ",\"Synthetic distributions match the WF-IoT artifact definitions; C++ and NumPy use different RNG streams.\""
        << ",\"For runs over one million queries, percentiles use the final one-million-query rolling window; mean uses all queries.\""
        << "]}";
    return out.str();
}

}  // namespace

std::string run_benchmark(const BenchmarkConfig& input, const TraceHooks& trace) {
    BenchmarkConfig c = input;
    validate(c);
    const std::vector<Method> order = parse_order(c.method_order);
    const std::vector<float> queries = generate_matrix(c.query_count, c.dimension,
                                                       c.seed, c.distribution, true);
    std::vector<MethodResult> results;
    results.reserve(4);
    for (Method method : order) results.push_back(run_method(method, c, queries, trace));

    const MethodResult* exact = nullptr;
    for (const MethodResult& result : results) {
        if (result.name == "exact_f32") exact = &result;
    }
    if (!exact) throw std::logic_error("exact result is missing");
    for (MethodResult& result : results) {
        result.recall = result.name == "exact_f32" ? 1.0 :
            recall_at_k(exact->first_ids, result.first_ids, c.query_count, c.k);
    }
    const IdleEnergyResult idle = measure_idle_energy(c, results, trace);
    return to_json(c, results, idle);
}

}  // namespace etar
