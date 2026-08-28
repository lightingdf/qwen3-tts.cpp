// BenchmarkHarness.swift
//
// On-device benchmark harness for VoiceEditor Issue #1 (iPad real-device
// sustained RTF / peak memory / thermal measurement for Qwen3-TTS-0.6B,
// Q4_0 GGUF, via the qwen3-tts.cpp GGML C++ engine).
//
// Backend selection is a RUNTIME concern, not a compile-time one. A single
// build of this app (linked against the Metal-enabled build-ios/ libs, which
// contain both the ggml-cpu and ggml-metal backends) is launched once per
// backend via:
//
//     xcrun devicectl device process launch \
//         -e '{"QWEN3_TTS_BACKEND": "cpu"}' \    # or "auto" for Metal-preferred
//         --terminate-existing <bundle-id>
//
// `QWEN3_TTS_BACKEND` is read directly by the C++ engine itself
// (src/gguf_loader.cpp, get_backend_mode_from_env()) — this Swift code does
// NOT set it and does not need to. Swift only *reads it back* from
// `ProcessInfo.processInfo.environment` purely to label the results file/
// records with whichever backend this particular process launch actually
// got (see `benchBackendLabel` below).
//
// This design is deliberate, not just "because the SA said so" — it was
// arrived at after a same-process contamination test
// (`/tmp/backend_switch_test.cpp`, macOS, 2026-08-28) showed that switching
// QWEN3_TTS_BACKEND mid-process across full load/destroy cycles measurably
// degrades subsequent runs (e.g. a same-process CPU run after a Metal run
// measured RTF 1.460, vs. ~1.09-1.17 for CPU measured in a fresh process;
// switching back to Metal afterward measured RTF 2.004, worse than that
// same process's own first Metal run at 1.625). The likely cause is Metal's
// residency-set `keep_alive` (180s) background management outliving the
// nominal `destroy()` call. Net effect: this harness NEVER switches backend
// in-process. Each backend's 5-run block is its own process launch, which
// also structurally satisfies the "insert a cooldown, don't run backends
// back-to-back" requirement below (the cooldown happens between two
// `devicectl launch` invocations, driven by the host-side runbook/script,
// not by in-app looping over both backends).
//
// The static-library CPU-only build variant (build-ios-cpu/, GGML_METAL=OFF)
// that was produced while investigating this is NOT what's used for the
// "cpu" backend measurements — it's kept only as an independent build-time
// cross-check. The actual iPad "cpu" runs use the *same* Metal-enabled
// binary as the "metal"/"auto" runs, forced to the CPU device at launch via
// QWEN3_TTS_BACKEND=cpu, because that's what the runtime-switchable-backend
// requirement (see Correction B below) actually asks for: one build, one
// switch, not two builds.
//
// ---------------------------------------------------------------------
// Build variants (set via Xcode build setting `SWIFT_ACTIVE_COMPILATION_
// CONDITIONS`):
//
//   Entitlement axis (which .entitlements file is attached to the target
//   at codesign time — this is a genuine build/codesign-time concern,
//   unlike backend, because a provisioning profile's capability set can't
//   be flipped by an environment variable at launch. This Swift code
//   cannot detect the entitlement at runtime, so the human wiring the
//   Xcode target MUST set this flag to match whichever entitlements/
//   *.entitlements file was actually applied):
//     - (default)                        -> ios-bench/entitlements/Bench-Baseline.entitlements
//     - BENCH_INCREASED_MEMORY_LIMIT     -> ios-bench/entitlements/Bench-IncreasedMemoryLimit.entitlements
//
// Bridging header: this target needs an Objective-C bridging header (Xcode
// build setting `SWIFT_OBJC_BRIDGING_HEADER`) pointing at
// ios-bench/BenchmarkHarness-Bridging-Header.h, and a header search path
// that includes ../src (for qwen3tts_c_api.h).
//
// ---------------------------------------------------------------------
// Operational notes (per PM review, 2026-08-28 — do not lose these when
// wiring this into an actual Xcode project and running on the iPad):
//
//  1. iPad Air 5 is passively cooled (no fan), unlike the M4 Mac mini used
//     for the initial macOS Metal-vs-CPU comparison. A backend that "wins"
//     on desktop under sustained load can lose on iPad once thermal
//     throttling kicks in mid-run. Do NOT assume the macOS CPU-faster-
//     than-Metal finding transfers to iPad without on-device measurement.
//  2. Because of (1), thermal state must be recorded as a *time series*
//     per run, not a single before/after/max snapshot — see
//     `RunMetrics.thermalSamples` below. Backend-vs-backend thermal
//     *behavior* (how fast each one heats the device, not just the final
//     RTF) is itself one of the things this benchmark exists to observe.
//  3. Never run the two backend blocks back-to-back, and never in the same
//     process (see the in-process contamination finding above — this is a
//     stronger constraint than the PM originally asked for, and subsumes
//     it). Insert an explicit cooldown between the two `devicectl launch`
//     invocations and confirm thermalState has returned to `.nominal`
//     before starting the next block. `cooldownBeforeBlock` exists for the
//     *operator* to invoke deliberately (e.g. from a thin runbook step) so
//     the wait and its outcome land in the same auditable JSONL file as the
//     runs it's protecting — it is not automatically chained between two
//     backends within one launch. Record however long the cooldown
//     actually took, and report it — don't just silently wait.
//  4. Record the actual `n_threads` used for every run. On macOS the
//     default (4 threads, incidentally equal to this Mac's performance-
//     core count) measurably beat a 10-thread run (RTF 1.091 vs 1.566) —
//     more threads made things *worse* for this batch=1 autoregressive
//     workload. Don't assume the same crossover point applies on iPad's
//     core layout; if time allows, run one default-thread-count vs one
//     P-core-count comparison per backend, but this is not a full
//     parameter search and shouldn't become one.
//  5. This harness records only what it directly measures. It does not
//     encode, and the eventual report must not cite, any RTF figure from
//     project docs (e.g. the architecture doc's RTF≈0.8 reference) as a
//     comparison target — that figure was measured on a different
//     software stack (MLX-Swift) and is not apples-to-apples with this
//     GGML C++ engine. That's a docs/ADR-level question for SA, not
//     something this harness should bake in an opinion about.
//  6. Every RunMetrics record carries both the phase-level wall clock
//     (`wallClockMs`) and that SAME run's own `audioDurationS` — RTF must
//     always be computed and reported from one run's own pair of numbers,
//     never a phase-time figure from one run divided by another run's
//     audio duration. (A cross-run mismatch of exactly this kind — RTF
//     computed against a different run's audio length than the phase
//     timings being cross-checked against it — was the source of an
//     apparent "self-inconsistent" 4.7s gap SA flagged in an earlier macOS
//     report; the underlying per-run numbers were consistent all along.)

import Foundation
#if canImport(Darwin)
import Darwin
#endif

/// Read back (never set) the backend this process was launched with, purely
/// for labeling results. The actual backend selection is performed by the
/// C++ engine (gguf_loader.cpp) reading this same environment variable
/// directly — see the file-level comment above for why this is launch-time
/// (via `devicectl ... -e`) rather than compile-time or in-process.
/// "auto" (Metal-preferred on Apple Silicon) is the engine's own default
/// when the variable is unset, so an absent env var is labeled "auto", not
/// left blank.
let benchBackendLabel: String = {
    let v = ProcessInfo.processInfo.environment["QWEN3_TTS_BACKEND"] ?? "auto"
    return v.lowercased()
}()

#if BENCH_INCREASED_MEMORY_LIMIT
let benchEntitlementLabel = "increased-memory-limit"
#else
let benchEntitlementLabel = "baseline"
#endif

// MARK: - Low-level memory / thermal sampling

/// `task_vm_info.phys_footprint` — the same figure Xcode's memory gauge and
/// iOS jetsam accounting use. Required by Issue #1.
func currentPhysFootprintBytes() -> UInt64 {
    var info = task_vm_info_data_t()
    var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
    let kr: kern_return_t = withUnsafeMutablePointer(to: &info) {
        $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
            task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
        }
    }
    guard kr == KERN_SUCCESS else { return 0 }
    return info.phys_footprint
}

/// `os_proc_available_memory()` — iOS 13+ headroom-to-jetsam-limit API.
/// Declared in <os/proc.h>, pulled in via the bridging header.
func currentOsProcAvailableMemoryBytes() -> Int {
    return Int(os_proc_available_memory())
}

func thermalStateString(_ s: ProcessInfo.ThermalState) -> String {
    switch s {
    case .nominal:  return "nominal"
    case .fair:     return "fair"
    case .serious:  return "serious"
    case .critical: return "critical"
    @unknown default: return "unknown"
    }
}

func thermalRank(_ s: ProcessInfo.ThermalState) -> Int {
    switch s {
    case .nominal:  return 0
    case .fair:     return 1
    case .serious:  return 2
    case .critical: return 3
    @unknown default: return 0
    }
}

// MARK: - Metrics types

struct ThermalSample: Codable {
    let elapsedMs: Double
    let thermalState: String
}

struct RunMetrics: Codable {
    let runIndex: Int
    let backend: String
    let entitlementVariant: String
    let nThreads: Int32
    var startedAt: String
    var completedAt: String?
    var wallClockMs: Double?
    var audioDurationS: Double?
    var rtf: Double?

    var physFootprintBeforeBytes: UInt64?
    var physFootprintAfterBytes: UInt64?
    var physFootprintPeakBytes: UInt64?

    var osAvailableMemoryBeforeBytes: Int?
    var osAvailableMemoryAfterBytes: Int?
    var osAvailableMemoryMinBytes: Int?

    var thermalStateBefore: String?
    var thermalStateAfter: String?
    /// Full per-run thermal time series (sampled at `samplingIntervalMs`
    /// alongside the memory samples below). Not just min/max: the PM
    /// explicitly asked for the curve because backend-vs-backend thermal
    /// *ramp behavior* is part of what this benchmark is meant to surface,
    /// not just the terminal value.
    var thermalSamples: [ThermalSample] = []

    var samplingIntervalMs: Double

    var success: Bool
    var errorMessage: String?
}

/// A block-level cooldown record, written between the two backends' 5-run
/// blocks so the report can show the device was actually back to a neutral
/// thermal state before the second block started (PM requirement: never
/// run the two backends back-to-back).
struct CooldownRecord: Codable {
    let beforeBackend: String
    let afterBackend: String
    let thermalStateAtCooldownStart: String
    let requestedCooldownSeconds: Double
    let actualCooldownSeconds: Double
    let thermalStateAtCooldownEnd: String
    let reachedNominal: Bool
}

// MARK: - JSONL result sink (crash/jetsam-kill-safe)

/// Append-only, fsync'd-per-write JSONL sink. Every event is flushed to
/// disk immediately so that if the process is jetsam-killed mid-run, the
/// events already written (including a `run_started` with no matching
/// `run_completed`) survive and are the forensic signal that a kill
/// happened mid-run — this is the mechanism Issue #1's OOM/jetsam-kill
/// detection requirement relies on. Do not batch/buffer writes.
final class ResultSink {
    private let handle: FileHandle
    private let encoder: JSONEncoder

    init(url: URL) {
        if !FileManager.default.fileExists(atPath: url.path) {
            FileManager.default.createFile(atPath: url.path, contents: nil)
        }
        guard let h = try? FileHandle(forWritingTo: url) else {
            fatalError("BenchmarkHarness: could not open results file at \(url)")
        }
        h.seekToEndOfFile()
        self.handle = h
        self.encoder = JSONEncoder()
    }

    func writeEvent(_ name: String, _ payload: [String: Any]) {
        var obj = payload
        obj["event"] = name
        obj["wallClockISO8601"] = isoNow()
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: []) else { return }
        handle.write(data)
        handle.write("\n".data(using: .utf8)!)
        try? handle.synchronize() // fsync — see class doc comment above
    }

    func writeCodable<T: Encodable>(_ name: String, _ value: T) {
        guard var dict = try? JSONSerialization.jsonObject(with: encoder.encode(value)) as? [String: Any] else {
            return
        }
        dict["event"] = name
        dict["wallClockISO8601"] = isoNow()
        guard let data = try? JSONSerialization.data(withJSONObject: dict, options: []) else { return }
        handle.write(data)
        handle.write("\n".data(using: .utf8)!)
        try? handle.synchronize()
    }

    func close() {
        try? handle.close()
    }
}

func isoNow() -> String {
    let f = ISO8601DateFormatter()
    f.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
    return f.string(from: Date())
}

// MARK: - Peak sampler (runs on a background thread while synth blocks)

/// Polls phys_footprint / os_proc_available_memory / thermalState on a
/// dedicated background thread while the (synchronous, blocking)
/// `qwen3_tts_synthesize` call runs on the calling thread. Records the full
/// thermal time series and the memory extremes observed during the call.
final class RunSampler {
    private let lock = NSLock()
    private var running = false
    private var thread: Thread?
    private let intervalMs: Double
    private let runStart = Date()

    private var peakPhys: UInt64 = 0
    private var minAvail: Int = .max
    private var samples: [ThermalSample] = []

    init(intervalMs: Double = 100) {
        self.intervalMs = intervalMs
    }

    func start() {
        lock.lock(); running = true; lock.unlock()
        let t = Thread { [weak self] in
            guard let self else { return }
            while true {
                self.lock.lock()
                let stillRunning = self.running
                self.lock.unlock()
                if !stillRunning { break }

                let p = currentPhysFootprintBytes()
                let a = currentOsProcAvailableMemoryBytes()
                let th = ProcessInfo.processInfo.thermalState
                let elapsed = Date().timeIntervalSince(self.runStart) * 1000

                self.lock.lock()
                if p > self.peakPhys { self.peakPhys = p }
                if a < self.minAvail { self.minAvail = a }
                self.samples.append(ThermalSample(elapsedMs: elapsed, thermalState: thermalStateString(th)))
                self.lock.unlock()

                Thread.sleep(forTimeInterval: self.intervalMs / 1000.0)
            }
        }
        t.stackSize = 256 * 1024
        thread = t
        t.start()
    }

    /// Stop sampling and return the collected extremes + full curve. Blocks
    /// briefly (<= one sampling interval) for the background thread to
    /// notice the stop request.
    func stop() -> (peakPhys: UInt64, minAvail: Int, samples: [ThermalSample]) {
        lock.lock(); running = false; lock.unlock()
        Thread.sleep(forTimeInterval: (intervalMs / 1000.0) + 0.05)
        lock.lock()
        let result = (peakPhys, minAvail, samples)
        lock.unlock()
        return result
    }
}

// MARK: - Benchmark driver

enum BenchmarkHarness {

    /// The confirmed 100-character Chinese benchmark text (see
    /// /tmp/bench_text_final.txt in the eng-1 working notes; duplicated
    /// here as a literal so the harness has no external file dependency at
    /// runtime beyond the model directory itself).
    static let benchText =
        "今天的天气非常晴朗，阳光洒满了整个城市的街道。人们纷纷走出家门，享受着难得的假期时光。" +
        "公园里孩子们在草坪上奔跑嬉戏，笑声传遍了每个角落。远处的湖面波光粼粼，几只白鹭掠过水面。" +
        "老人们坐着闲聊，笑容满面。"

    static let numRuns = 5
    static let samplingIntervalMs = 100.0

    /// Directory containing qwen3-tts-0.6b-f16.gguf (actually Q4_0 tensors
    /// under that historical filename — see PR description for why) and
    /// qwen3-tts-tokenizer-f16.gguf. Models are staged into the app's
    /// Documents directory (e.g. via `devicectl device copy-to`) rather
    /// than bundled as app resources, to avoid bloating the signed app
    /// bundle with ~1.4GB of weights.
    static var modelDirURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("models", isDirectory: true)
    }

    static var resultsFileURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("bench_results_\(benchBackendLabel)_\(benchEntitlementLabel).jsonl")
    }

    /// Wait for the device to cool down before starting a fresh backend's
    /// 5-run block. Polls thermalState every second; gives up (and records
    /// `reachedNominal: false`) after `maxWaitSeconds` so a stuck-`.fair`
    /// device doesn't hang the harness forever — but ALWAYS records what
    /// actually happened so the report can flag a contaminated block.
    static func cooldownBeforeBlock(previousBackend: String,
                                     nextBackend: String,
                                     minWaitSeconds: Double = 120,
                                     maxWaitSeconds: Double = 600,
                                     sink: ResultSink) {
        let startState = ProcessInfo.processInfo.thermalState
        let startTime = Date()
        // Always wait at least minWaitSeconds regardless of thermalState,
        // then keep waiting (up to maxWaitSeconds) until nominal.
        while Date().timeIntervalSince(startTime) < minWaitSeconds {
            Thread.sleep(forTimeInterval: 1.0)
        }
        while ProcessInfo.processInfo.thermalState != .nominal &&
              Date().timeIntervalSince(startTime) < maxWaitSeconds {
            Thread.sleep(forTimeInterval: 1.0)
        }
        let endState = ProcessInfo.processInfo.thermalState
        let actual = Date().timeIntervalSince(startTime)
        let record = CooldownRecord(
            beforeBackend: previousBackend,
            afterBackend: nextBackend,
            thermalStateAtCooldownStart: thermalStateString(startState),
            requestedCooldownSeconds: minWaitSeconds,
            actualCooldownSeconds: actual,
            thermalStateAtCooldownEnd: thermalStateString(endState),
            reachedNominal: endState == .nominal
        )
        sink.writeCodable("cooldown", record)
    }

    /// Runs the configured number of generations against a single already-
    /// loaded engine handle and appends one `run_started` + one
    /// `run_completed` (or `run_failed`) event per run to `sink`.
    static func runBlock(nThreads: Int32, sink: ResultSink) {
        var params = Qwen3TtsParams()
        qwen3_tts_default_params(&params)
        params.language_id = 2055 // zh
        params.n_threads = nThreads

        let modelDirPath = modelDirURL.path
        guard let tts = modelDirPath.withCString({ qwen3_tts_create($0, nThreads) }) else {
            sink.writeEvent("fatal", ["message": "qwen3_tts_create failed for \(modelDirPath)"])
            return
        }
        defer { qwen3_tts_destroy(tts) }

        for i in 1...numRuns {
            var metrics = RunMetrics(
                runIndex: i,
                backend: benchBackendLabel,
                entitlementVariant: benchEntitlementLabel,
                nThreads: nThreads,
                startedAt: isoNow(),
                samplingIntervalMs: samplingIntervalMs,
                success: false
            )
            // Written BEFORE generation starts: if the process is
            // jetsam-killed mid-run, this is the last event on disk for
            // run i, with no matching run_completed — that absence is the
            // OOM/jetsam-kill signal Issue #1 asks us to detect.
            sink.writeCodable("run_started", metrics)

            let beforePhys = currentPhysFootprintBytes()
            let beforeAvail = currentOsProcAvailableMemoryBytes()
            let beforeThermal = ProcessInfo.processInfo.thermalState

            let sampler = RunSampler(intervalMs: samplingIntervalMs)
            sampler.start()
            let t0 = Date()

            let audio = benchText.withCString { textC -> UnsafeMutablePointer<Qwen3TtsAudio>? in
                withUnsafePointer(to: params) { paramsPtr in
                    qwen3_tts_synthesize(tts, textC, paramsPtr)
                }
            }

            let elapsedMs = Date().timeIntervalSince(t0) * 1000
            let (peakPhys, minAvail, samples) = sampler.stop()

            metrics.completedAt = isoNow()
            metrics.wallClockMs = elapsedMs
            metrics.physFootprintBeforeBytes = beforePhys
            metrics.physFootprintAfterBytes = currentPhysFootprintBytes()
            metrics.physFootprintPeakBytes = max(peakPhys, beforePhys)
            metrics.osAvailableMemoryBeforeBytes = beforeAvail
            metrics.osAvailableMemoryAfterBytes = currentOsProcAvailableMemoryBytes()
            metrics.osAvailableMemoryMinBytes = min(minAvail, beforeAvail)
            metrics.thermalStateBefore = thermalStateString(beforeThermal)
            metrics.thermalStateAfter = thermalStateString(ProcessInfo.processInfo.thermalState)
            metrics.thermalSamples = samples

            if let audio = audio {
                let audioDurationS = Double(audio.pointee.n_samples) / Double(audio.pointee.sample_rate)
                metrics.audioDurationS = audioDurationS
                metrics.rtf = (elapsedMs / 1000.0) / audioDurationS
                metrics.success = true
                qwen3_tts_free_audio(audio)
            } else {
                metrics.success = false
                metrics.errorMessage = String(cString: qwen3_tts_get_error(tts))
            }

            sink.writeCodable("run_completed", metrics)
        }
    }

    static var cooldownResultsFileURL: URL {
        FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("bench_cooldown.jsonl")
    }

    /// Entry point — call this from the app target once wired up (e.g.
    /// from `applicationDidFinishLaunching` or a SwiftUI `.task {}`).
    ///
    /// Dispatches on `QWEN3_TTS_BENCH_MODE` (default "run") so that the
    /// *cooldown* step between two backends' blocks can also be driven
    /// purely by separate `devicectl process launch` invocations with
    /// different env vars — never by looping over both backends inside one
    /// process (that was tried and found to contaminate results; see the
    /// file-level comment). The intended host-side sequence for one
    /// backend transition is three launches of this same app:
    ///
    ///   1. `-e '{"QWEN3_TTS_BACKEND":"auto"}'`
    ///        -> mode=run (default): 5-run Metal/auto block
    ///   2. `-e '{"QWEN3_TTS_BENCH_MODE":"cooldown",
    ///           "QWEN3_TTS_BENCH_PREV_BACKEND":"auto",
    ///           "QWEN3_TTS_BENCH_NEXT_BACKEND":"cpu"}'`
    ///        -> waits for thermalState to return to nominal (or times out),
    ///           records one CooldownRecord to bench_cooldown.jsonl, exits
    ///   3. `-e '{"QWEN3_TTS_BACKEND":"cpu"}'`
    ///        -> mode=run: 5-run CPU block
    ///
    /// See phase0/ipad-bench/scripts/run_bench.sh (parent repo) for the
    /// actual host-side orchestration that issues these three launches.
    static func run() {
        let env = ProcessInfo.processInfo.environment
        let mode = (env["QWEN3_TTS_BENCH_MODE"] ?? "run").lowercased()

        if mode == "cooldown" {
            let prev = (env["QWEN3_TTS_BENCH_PREV_BACKEND"] ?? "unknown").lowercased()
            let next = (env["QWEN3_TTS_BENCH_NEXT_BACKEND"] ?? "unknown").lowercased()
            let sink = ResultSink(url: cooldownResultsFileURL)
            cooldownBeforeBlock(previousBackend: prev, nextBackend: next, sink: sink)
            sink.close()
            return
        }

        let sink = ResultSink(url: resultsFileURL)
        sink.writeEvent("session_started", [
            "backend": benchBackendLabel,
            "entitlementVariant": benchEntitlementLabel,
            "numRuns": numRuns,
            "benchTextLength": benchText.count,
            // Read at launch, before qwen3_tts_create / any model load, so
            // this is available even if no model is staged yet. This is the
            // decisive baseline/increased-memory-limit comparison point: if
            // the increased-memory-limit entitlement did not actually take
            // effect (e.g. a provisioning-profile/signing-certificate team
            // mismatch caused the kernel to silently ignore it), this number
            // will be indistinguishable between the two entitlement variants
            // even though the entitlement is present in the signed binary.
            "osAvailableMemoryAtLaunchBytes": currentOsProcAvailableMemoryBytes(),
            "physFootprintAtLaunchBytes": currentPhysFootprintBytes(),
            "physicalMemoryBytes": ProcessInfo.processInfo.physicalMemory,
        ])

        // Default thread count mirrors qwen3-tts.cpp's own CLI default (4).
        // Per PM guidance: at most one extra comparison point (P-core
        // count), not a search. Use activeProcessorCount as that second
        // data point when this constant is flipped by hand for a one-off
        // comparison run; do not wire up multiple in-app configurations.
        let nThreads: Int32 = 4
        runBlock(nThreads: nThreads, sink: sink)

        sink.writeEvent("session_completed", [:])
        sink.close()
    }
}
