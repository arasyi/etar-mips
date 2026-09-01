#pragma once

#include <cstdint>
#include <functional>
#include <string>

namespace etar {

struct BenchmarkConfig {
    int64_t corpus_size = 50000;
    int32_t dimension = 256;
    int32_t query_count = 1000;
    int32_t warmup_queries = 100;
    int32_t repetitions = 1;
    int32_t k = 10;
    int32_t rerank = 100;
    float rho = 0.90f;
    int32_t head_cap = 128;
    float lambda_tail = 0.75f;
    float alpha_min = 0.05f;
    float alpha_max = 0.50f;
    int64_t seed = 123;
    float measurement_seconds = 10.0f;
    bool quantize_query = false;
    int32_t cpu_core = -1;
    std::string distribution = "dense_gaussian";
    std::string method_order = "exact,full_i8,etar,etar_lm";
};

struct TraceHooks {
    std::function<void(const char*)> begin;
    std::function<void()> end;
    // Returns remaining battery energy in joules, or NaN when unavailable.
    std::function<double()> energy_joules;
};

// Runs exact float32, exhaustive int8, ETAR, and ETAR-LM on identical data.
// The returned string is a complete JSON document.
std::string run_benchmark(const BenchmarkConfig& config,
                          const TraceHooks& trace = {});

}  // namespace etar
