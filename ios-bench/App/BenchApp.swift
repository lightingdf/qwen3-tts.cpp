// BenchApp.swift
//
// Minimal SwiftUI app shell for VoiceEditor's on-device benchmark (issue #1
// on iPad, issue #4 on iPhone — same harness, same two targets, device
// picked at launch time via `devicectl`'s --device UDID, not a build axis).
// This target has no product purpose beyond driving BenchmarkHarness.run()
// on a real, physically-present device — see ios-bench/BenchmarkHarness.swift
// for the actual measurement logic and phase0/ipad-bench/README.md (parent
// repo) for how it's launched via `xcrun devicectl`.
//
// Two things this file is responsible for, beyond just calling run():
//
// 1. Not blocking the main thread. BenchmarkHarness.run() is a long-running
//    synchronous call chain (two 5-run blocks, each run doing full
//    tokenize -> transformer -> decode synthesis on the calling thread).
//    Doing that on the main thread would freeze the UI and risk the
//    springboard/UIKit "app unresponsive" watchdog killing the process
//    before the benchmark finishes. Dispatching to a detached thread avoids
//    that; RunSampler's own background sampling thread is unaffected either
//    way.
//
// 2. Disabling the idle timer. If the device auto-locks mid-run, this
//    process can be suspended, silently truncating a "sustained load"
//    measurement that issue #1 specifically asks for (sustained RTF across
//    multiple runs, not just a single cold-start number). This is an
//    operational correctness requirement for the benchmark, not a product
//    feature.
import SwiftUI
import UIKit

@main
struct BenchApp: App {
    @UIApplicationDelegateAdaptor(BenchAppDelegate.self) var delegate

    var body: some Scene {
        WindowGroup {
            BenchStatusView()
        }
    }
}

final class BenchAppDelegate: NSObject, UIApplicationDelegate {
    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil
    ) -> Bool {
        application.isIdleTimerDisabled = true

        Thread.detachNewThread {
            BenchmarkHarness.run()
        }

        return true
    }
}

struct BenchStatusView: View {
    var body: some View {
        VStack(spacing: 12) {
            Text("VoiceEditor Benchmark (\(UIDevice.current.model))")
                .font(.headline)
            Text("backend=\(benchBackendLabel)  entitlement=\(benchEntitlementLabel)")
                .font(.caption)
                .foregroundStyle(.secondary)
            Text("Running in the background for issue #1. Do not lock the screen or switch away from this app until the launch completes — devicectl will relaunch it for the next backend/cooldown step.")
                .font(.caption2)
                .multilineTextAlignment(.center)
                .padding(.horizontal)
        }
        .padding()
    }
}
