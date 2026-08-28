// BenchmarkHarness-Bridging-Header.h
//
// Objective-C bridging header for the Issue #1 on-device benchmark app
// target. Wire this in via the Xcode build setting
// `SWIFT_OBJC_BRIDGING_HEADER = ios-bench/BenchmarkHarness-Bridging-Header.h`,
// and add `../src` to Header Search Paths so `qwen3tts_c_api.h` resolves.
//
// This header intentionally does NOT modify or wrap qwen3tts_c_api.h — the
// benchmark harness only calls the existing, already-tested C API. See
// BenchmarkHarness.swift for why memory/thermal instrumentation is done
// entirely on the Swift/OS side (task_vm_info, os_proc_available_memory,
// ProcessInfo.thermalState) instead of extending the C++ engine's own
// internal tts_result timing/memory fields, which the C API does not
// currently expose.
#ifndef BENCHMARK_HARNESS_BRIDGING_HEADER_H
#define BENCHMARK_HARNESS_BRIDGING_HEADER_H

#include "qwen3tts_c_api.h"
#include <os/proc.h>

#endif /* BENCHMARK_HARNESS_BRIDGING_HEADER_H */
