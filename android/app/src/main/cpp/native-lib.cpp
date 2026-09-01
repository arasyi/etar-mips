#include "etar_core.h"

#include <android/trace.h>
#include <jni.h>

#include <cerrno>
#include <exception>
#include <limits>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <system_error>

extern "C" JNIEXPORT jstring JNICALL
Java_ae_kunet_etarbenchmark_MainActivity_runNativeBenchmark(
        JNIEnv* env, jobject activity,
        jlong corpus_size, jint dimension, jint query_count,
        jint warmup_queries, jint repetitions, jint k, jint rerank,
        jfloat rho, jint head_cap, jfloat lambda_tail,
        jfloat alpha_min, jfloat alpha_max, jlong seed,
        jfloat measurement_seconds, jboolean quantize_query, jint cpu_core,
        jstring distribution, jstring method_order) {
    const char* dist_chars = env->GetStringUTFChars(distribution, nullptr);
    const char* order_chars = env->GetStringUTFChars(method_order, nullptr);
    cpu_set_t original_affinity;
    bool restore_affinity = false;
    try {
        etar::BenchmarkConfig c;
        c.corpus_size = corpus_size;
        c.dimension = dimension;
        c.query_count = query_count;
        c.warmup_queries = warmup_queries;
        c.repetitions = repetitions;
        c.k = k;
        c.rerank = rerank;
        c.rho = rho;
        c.head_cap = head_cap;
        c.lambda_tail = lambda_tail;
        c.alpha_min = alpha_min;
        c.alpha_max = alpha_max;
        c.seed = seed;
        c.measurement_seconds = measurement_seconds;
        c.quantize_query = quantize_query == JNI_TRUE;
        c.cpu_core = cpu_core;
        c.distribution = dist_chars;
        c.method_order = order_chars;
        env->ReleaseStringUTFChars(distribution, dist_chars);
        env->ReleaseStringUTFChars(method_order, order_chars);
        dist_chars = nullptr;
        order_chars = nullptr;

        const bool have_original =
            sched_getaffinity(0, sizeof(original_affinity), &original_affinity) == 0;
        if (c.cpu_core >= 0) {
            if (!have_original) {
                throw std::system_error(errno, std::generic_category(),
                                        "could not read benchmark thread affinity");
            }
            if (c.cpu_core >= CPU_SETSIZE) {
                throw std::invalid_argument("requested CPU core exceeds CPU_SETSIZE");
            }
            cpu_set_t requested;
            CPU_ZERO(&requested);
            CPU_SET(c.cpu_core, &requested);
            if (sched_setaffinity(0, sizeof(requested), &requested) != 0) {
                throw std::system_error(errno, std::generic_category(),
                                        "could not pin benchmark thread");
            }
            restore_affinity = true;
        }
        etar::TraceHooks trace;
        trace.begin = [](const char* name) { ATrace_beginSection(name); };
        trace.end = []() { ATrace_endSection(); };
        jclass activity_class = env->GetObjectClass(activity);
        jmethodID energy_method = env->GetMethodID(
            activity_class, "readBatteryEnergyNwh", "()J");
        if (energy_method == nullptr) {
            throw std::runtime_error("could not resolve battery energy callback");
        }
        trace.energy_joules = [env, activity, energy_method]() {
            const jlong nwh = env->CallLongMethod(activity, energy_method);
            if (nwh == std::numeric_limits<jlong>::min()) {
                return std::numeric_limits<double>::quiet_NaN();
            }
            return static_cast<double>(nwh) * 3.6e-6;
        };
        const std::string result = etar::run_benchmark(c, trace);
        if (restore_affinity) {
            sched_setaffinity(0, sizeof(original_affinity), &original_affinity);
            restore_affinity = false;
        }
        return env->NewStringUTF(result.c_str());
    } catch (const std::exception& e) {
        if (restore_affinity) {
            sched_setaffinity(0, sizeof(original_affinity), &original_affinity);
        }
        if (dist_chars) env->ReleaseStringUTFChars(distribution, dist_chars);
        if (order_chars) env->ReleaseStringUTFChars(method_order, order_chars);
        jclass type = env->FindClass("java/lang/RuntimeException");
        env->ThrowNew(type, e.what());
        return nullptr;
    }
}
