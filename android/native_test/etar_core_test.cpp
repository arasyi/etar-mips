#include "etar_core.h"

#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    etar::BenchmarkConfig c;
    c.corpus_size = 512;
    c.dimension = 32;
    c.query_count = 4;
    c.warmup_queries = 1;
    c.repetitions = 1;
    c.k = 5;
    c.rerank = 64;
    c.head_cap = 16;
    c.measurement_seconds = 0.0f;
    const char* distributions[] = {
        "dense_gaussian", "sparse10", "mixed_dropout_heavytail", "norm_heavy"
    };
    const char* orders[] = {
        "exact,full_i8,etar,etar_lm",
        "full_i8,etar,etar_lm,exact",
        "etar,etar_lm,exact,full_i8",
        "etar_lm,exact,full_i8,etar"
    };
    for (int variant = 0; variant < 4; ++variant) {
        c.measurement_seconds = variant == 0 ? 0.001f : 0.0f;
        c.distribution = distributions[variant];
        c.quantize_query = variant == 1;
        c.method_order = orders[variant];
        etar::TraceHooks hooks;
        double remaining_joules = 100.0;
        if (variant == 0) {
            hooks.energy_joules = [&remaining_joules]() {
                return remaining_joules--;
            };
        }
        const std::string result = etar::run_benchmark(c, hooks);
        if (result.find("\"exact_f32\"") == std::string::npos ||
            result.find("\"full_i8_scan\"") == std::string::npos ||
            result.find("\"etar_lm\"") == std::string::npos ||
            result.find("\"recall_at_k\"") == std::string::npos ||
            (variant == 0 && result.find("\"energy_counter_status\":\"measured\"") ==
                std::string::npos) ||
            (variant == 0 && result.find("\"idle_energy_baseline\":{\"status\":\"measured\"") ==
                std::string::npos) ||
            result.find("\"distribution\":\"" + c.distribution + "\"") == std::string::npos) {
            std::cerr << result << '\n';
            return 1;
        }
        if (variant == 0) std::cout << result << '\n';
    }

    c.rho = 0.0f;
    try {
        etar::run_benchmark(c);
        std::cerr << "invalid rho was accepted\n";
        return 1;
    } catch (const std::invalid_argument&) {
        // Expected.
    }
    return 0;
}
