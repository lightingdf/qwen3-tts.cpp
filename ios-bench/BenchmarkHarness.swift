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
//
// ---------------------------------------------------------------------
// Second round of PM+SA review, 2026-08-28 (systemic AC gap found across
// ALL Phase 0 issues, not just this one — issue #1's AC was rewritten to
// encode these two formally; re-pull via `gh issue view 1` if unsure):
//
//  7. "success" (qwen3_tts_synthesize returned a non-null pointer) is NOT
//     evidence the output is correct — a broken config can produce fluent-
//     length garbage, wrong-language output, or a truncated/looped passage
//     while still returning a clean, non-null result (this fork's own git
//     history has a real precedent: a broken Q4_K quantization conversion
//     that exited 0 and produced a non-null "garbage" inference result).
//     `evaluateOutputSanity` below is only the CHEAP, first tier of a
//     three-tier validity gate; it cannot catch "fluent but wrong". The
//     second tier is a host-side ASR round-trip (sherpa-onnx SenseVoice,
//     reused as-is from issue #2 — see phase0/ipad-bench/scripts/
//     check_asr_validity.py in the parent repo) computing CER against the
//     known input text for the WAV saved on run 1 of each block; the third
//     tier is a human actually listening to that same WAV. Per the AC: if
//     ANY of the three tiers fails for a given backend's block, that
//     backend's ENTIRE RTF/memory dataset is INVALID and must not be cited
//     as acceptance evidence — this is a per-backend verdict, not per-run.
//  8. Model LOAD time (mmap + engine init, i.e. `qwen3_tts_create`) must be
//     measured separately from generation time. The user-facing "100 字 ≤
//     2 分钟" commitment is actual wait time from button-press to hearing
//     audio, which for a first-time-this-session synthesis includes model
//     load — reporting generation-only would understate real user wait.
//     `runBlock` below times `qwen3_tts_create` and records it as both a
//     dedicated `model_loaded` event (written immediately, so it survives
//     even if run 1 itself never completes) and as `RunMetrics.modelLoadMs`
//     on run 1's own record (for convenient colocated reporting). The
//     eventual report must present BOTH cold-path-total (modelLoadMs +
//     run 1's wallClockMs — the real first-use wait) and warm-path-only
//     (runs 2-5's wallClockMs, model already resident) side by side, and
//     state explicitly which one the "≤2 min" pass/fail verdict is judged
//     against.

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

    // --- Output sanity fields (PM review, 2026-08-28) ---------------------
    // `success` above only ever meant "qwen3_tts_synthesize returned a
    // non-null pointer" — it says nothing about whether the audio behind
    // that pointer is actually usable speech. A model that emits 22s of
    // noise, silence, or truncated garbage would still report a clean RTF
    // and success=true with no other signal. These fields are cheap
    // (single-pass array scans over already-in-memory float samples) sanity
    // checks computed for EVERY run, not just the one that gets saved as a
    // WAV, specifically so a report can't accidentally present "fast" as
    // "usable" without evidence.
    var nSamples: Int32?
    var sampleRate: Int32?
    var rmsAmplitude: Double?
    var peakAmplitude: Double?
    var clippedSampleFraction: Double?
    var outputSanityPassed: Bool?
    var outputSanityNotes: [String]?
    /// Set only for the one run per block (run 1) whose audio is dumped to
    /// a WAV file for a human (the PM) to actually listen to and confirm
    /// it's intelligible Mandarin — see PM's 2026-08-28 review: RTF/memory
    /// numbers are not evidence the approach is viable until someone has
    /// heard the output. Writing this file happens strictly AFTER
    /// wallClockMs above is already captured, so file I/O never contaminates
    /// the timed measurement.
    var savedWavPath: String?

    /// Set only on run 1 of each block (see PM review, 2026-08-28, note 8 in
    /// the file header): time spent in `qwen3_tts_create` (mmap + engine
    /// init), measured immediately before this loop starts and BEFORE any
    /// per-run timing below, so it never contaminates any run's own
    /// `wallClockMs`. Also emitted redundantly as a standalone
    /// `model_loaded` event (see `runBlock`) so the figure survives on disk
    /// even if run 1 itself is jetsam-killed before `run_completed` is
    /// written. Cold-path-total (the real first-use wait the "100 字 ≤ 2
    /// 分钟" commitment is about) = this + run 1's own `wallClockMs`;
    /// warm-path = runs 2-5's `wallClockMs` with the model already resident.
    var modelLoadMs: Double?

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

// MARK: - Output sanity checks & WAV dump (PM review, 2026-08-28)
//
// `qwen3_tts_synthesize` returning a non-null pointer only proves the call
// didn't crash/error out — it says nothing about whether what's behind that
// pointer is intelligible Mandarin speech, silence, noise, or a truncated
// fragment. Issue #1 exists to answer "is this model USABLE on-device", not
// "does this function return quickly", so every run gets a few cheap,
// automatable checks on top of the timing/memory numbers, and the first run
// of each block gets dumped to a WAV a human can actually listen to.

/// Result of scanning one run's raw float32 samples. All three checks are
/// single-pass, O(n) array scans over data that's already resident in
/// memory (the `Qwen3TtsAudio` this harness already holds a pointer to) —
/// negligible cost relative to a multi-second synthesis call.
struct OutputSanityCheck {
    let rms: Double
    let peak: Double
    let clippedFraction: Double
    let passed: Bool
    let notes: [String]
}

/// `expectedTextLength` is the benchmark text's character count (currently
/// always ~100 Chinese characters — see `BenchmarkHarness.benchText`).
/// Issue #1's AC (re-pulled 2026-08-28, per PM+SA joint review) specifies
/// ~100 Chinese characters should land around 25-35s of audio, "明显偏离即
/// 无效" (an obvious deviation is invalid) — the bounds below give that a
/// generous buffer on both sides (not a strict spec, and deliberately
/// looser than 25-35s itself) so this only fires on a GROSS failure
/// (near-total silence, runaway/truncated generation), not on ordinary
/// run-to-run variance. NOTE: this duration check is only tier (a) of a
/// three-tier validity gate (see file header, note 7) — it cannot by
/// itself catch "fluent-length but wrong" output (wrong language, wrong
/// passage); that's what the ASR round-trip CER check (tier b, host-side,
/// phase0/ipad-bench/scripts/check_asr_validity.py) and human listening
/// (tier c) are for.
func evaluateOutputSanity(
    samples: UnsafeBufferPointer<Float>,
    sampleRate: Int32,
    audioDurationS: Double,
    expectedTextLength: Int
) -> OutputSanityCheck {
    var notes: [String] = []

    let minExpectedS = 15.0
    let maxExpectedS = 45.0
    if audioDurationS < minExpectedS || audioDurationS > maxExpectedS {
        notes.append("audioDurationS=\(audioDurationS) outside expected [\(minExpectedS), \(maxExpectedS)]s for a ~\(expectedTextLength)-character Mandarin passage")
    }

    var sumSq: Double = 0
    var peak: Double = 0
    var clippedCount = 0
    let clipThreshold: Float = 0.995
    let n = samples.count
    for v in samples {
        let av = Double(abs(v))
        sumSq += av * av
        if av > peak { peak = av }
        if abs(v) >= clipThreshold { clippedCount += 1 }
    }
    let rms = n > 0 ? (sumSq / Double(n)).squareRoot() : 0
    let clippedFraction = n > 0 ? Double(clippedCount) / Double(n) : 0

    let silenceRmsThreshold = 0.001 // roughly -60 dBFS
    if rms < silenceRmsThreshold {
        notes.append("rms=\(rms) at/below silence threshold \(silenceRmsThreshold) — output is likely silence, not speech")
    }

    let clippingFractionThreshold = 0.01 // >1% of samples pinned near full-scale
    if clippedFraction > clippingFractionThreshold {
        notes.append("clippedFraction=\(clippedFraction) exceeds \(clippingFractionThreshold) — output may be clipped/distorted")
    }

    return OutputSanityCheck(rms: rms, peak: peak, clippedFraction: clippedFraction, passed: notes.isEmpty, notes: notes)
}

/// Minimal, dependency-free 16-bit PCM mono WAV writer. This exists purely
/// so the PM can listen to real on-device output and confirm the ~100
/// Chinese characters were actually rendered as intelligible speech — it is
/// evidence for a human ear, not part of the measurement, and (per the
/// caller in `runBlock`) is only ever invoked AFTER `wallClockMs` has
/// already been captured so this I/O never contaminates the timed run.
func writeWav(samples: UnsafeBufferPointer<Float>, sampleRate: Int32, to url: URL) throws {
    let n = samples.count
    var pcm16 = [Int16](repeating: 0, count: n)
    for i in 0..<n {
        let clamped = max(-1.0, min(1.0, Double(samples[i])))
        pcm16[i] = Int16(clamped * Double(Int16.max))
    }

    let dataSize = UInt32(n * MemoryLayout<Int16>.size)
    let byteRate = UInt32(sampleRate) * UInt32(MemoryLayout<Int16>.size)
    let blockAlign = UInt16(MemoryLayout<Int16>.size)

    func le<T: FixedWidthInteger>(_ v: T) -> [UInt8] {
        withUnsafeBytes(of: v.littleEndian) { Array($0) }
    }

    var header = Data()
    header.append(contentsOf: Array("RIFF".utf8))
    header.append(contentsOf: le(UInt32(36) + dataSize))
    header.append(contentsOf: Array("WAVE".utf8))
    header.append(contentsOf: Array("fmt ".utf8))
    header.append(contentsOf: le(UInt32(16)))          // fmt chunk size
    header.append(contentsOf: le(UInt16(1)))           // PCM
    header.append(contentsOf: le(UInt16(1)))           // mono
    header.append(contentsOf: le(UInt32(sampleRate)))
    header.append(contentsOf: le(byteRate))
    header.append(contentsOf: le(blockAlign))
    header.append(contentsOf: le(UInt16(16)))          // bits per sample
    header.append(contentsOf: Array("data".utf8))
    header.append(contentsOf: le(dataSize))

    var full = header
    pcm16.withUnsafeBufferPointer { buf in
        full.append(Data(buffer: buf))
    }
    try full.write(to: url, options: .atomic)
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
        // Timed separately from every run below (PM review, 2026-08-28, note
        // 8 in the file header): this is mmap + engine init, not generation.
        // The "100 字 ≤ 2 分钟" user-facing commitment is actual wait time
        // from button-press to hearing audio, which for a first-time-this-
        // session synthesis includes this load time — folding it silently
        // into run 1's own wallClockMs (or omitting it entirely, as before
        // this review) would either double-count it into "generation" or
        // drop it from the report altogether. Neither is acceptable, so it
        // gets its own clock and its own field/event.
        let modelLoadT0 = Date()
        guard let tts = modelDirPath.withCString({ qwen3_tts_create($0, nThreads) }) else {
            sink.writeEvent("fatal", ["message": "qwen3_tts_create failed for \(modelDirPath)"])
            return
        }
        let modelLoadMs = Date().timeIntervalSince(modelLoadT0) * 1000
        // Written immediately (not deferred to run 1's run_completed) so this
        // figure is on disk even if run 1 itself is jetsam-killed before it
        // finishes — see ResultSink's doc comment on why events are flushed
        // per-write rather than batched.
        sink.writeEvent("model_loaded", [
            "backend": benchBackendLabel,
            "entitlementVariant": benchEntitlementLabel,
            "nThreads": nThreads,
            "modelLoadMs": modelLoadMs,
        ])
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
            if i == 1 {
                // See file header note 8 / the `model_loaded` event above:
                // colocating this on run 1's own record too (in addition to
                // the standalone event) is purely for report convenience —
                // the standalone event is the authoritative/robust copy.
                metrics.modelLoadMs = modelLoadMs
            }
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
                let nSamples = audio.pointee.n_samples
                let sampleRate = audio.pointee.sample_rate
                let audioDurationS = Double(nSamples) / Double(sampleRate)
                metrics.audioDurationS = audioDurationS
                metrics.rtf = (elapsedMs / 1000.0) / audioDurationS
                metrics.nSamples = nSamples
                metrics.sampleRate = sampleRate

                // Everything below reads the still-valid `audio.pointee.samples`
                // buffer but happens AFTER wallClockMs was already captured
                // above, so none of it — sanity scan or WAV write — can
                // contaminate the timed measurement (PM requirement).
                let samplesBuffer = UnsafeBufferPointer(start: audio.pointee.samples, count: Int(nSamples))
                let sanity = evaluateOutputSanity(
                    samples: samplesBuffer,
                    sampleRate: sampleRate,
                    audioDurationS: audioDurationS,
                    expectedTextLength: benchText.count
                )
                metrics.rmsAmplitude = sanity.rms
                metrics.peakAmplitude = sanity.peak
                metrics.clippedSampleFraction = sanity.clippedFraction
                metrics.outputSanityPassed = sanity.passed
                metrics.outputSanityNotes = sanity.notes

                // Only run 1 of each block gets saved as a WAV: enough for a
                // human to confirm intelligibility without paying repeated
                // disk I/O across all 5 runs (which would also risk
                // polluting later runs' "before" memory/thermal baselines
                // with this run's file-write activity).
                if i == 1 {
                    let wavURL = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
                        .appendingPathComponent("sample_\(benchBackendLabel)_\(benchEntitlementLabel).wav")
                    do {
                        try writeWav(samples: samplesBuffer, sampleRate: sampleRate, to: wavURL)
                        metrics.savedWavPath = wavURL.path
                    } catch {
                        sink.writeEvent("wav_write_failed", [
                            "runIndex": i,
                            "path": wavURL.path,
                            "error": String(describing: error),
                        ])
                    }
                }

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
            // `run_bench.sh` launches this app via `devicectl device process
            // launch --console`, which blocks until the process actually
            // exits (confirmed via `devicectl ... launch --help`). This app
            // never returns to a foreground UI state worth keeping alive —
            // its only job per launch is "do the work, flush the JSONL,
            // exit" — so without this call the host script would hang here
            // forever waiting for a termination that never happens on its
            // own (the SwiftUI WindowGroup keeps the process alive
            // indefinitely otherwise).
            exit(0)
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
        // See the matching comment in the mode=="cooldown" branch above:
        // `--console`-attached devicectl launches block until this process
        // exits, and nothing else in this app ever terminates it.
        exit(0)
    }
}
