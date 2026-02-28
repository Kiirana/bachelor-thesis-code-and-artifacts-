//
//  LatencyTracker.swift
//  CurbDetectorApp
//
//  Per-stage latency measurement using CACurrentMediaTime() for thesis evaluation.
//
//  Reference:
//    Apple Developer Documentation – CACurrentMediaTime()
//    https://developer.apple.com/documentation/quartzcore/1395
//
//  Usage:
//    let tracker = LatencyTracker()
//    tracker.start("yolo_detection")
//    // ... run detection ...
//    tracker.stop("yolo_detection")
//    let result = tracker.pipelineResult()
//    print(result.totalMs)  // end-to-end in milliseconds
//

import QuartzCore   // CACurrentMediaTime()
import Foundation

// MARK: - Single-stage timing result

/// One measured stage of the inference pipeline.
struct StageLatency: Codable, Identifiable {
    let id: String           // stage name (e.g. "yolo_detection")
    let durationMs: Double   // wall-clock milliseconds
}

// MARK: - Full pipeline timing result

/// All stages for a single image inference, plus the computed total.
struct PipelineLatency: Codable {
    let stages: [StageLatency]
    let totalMs: Double
}

// MARK: - Aggregated statistics for thesis tables

/// Mean / std / min / max / count over N pipeline runs.
struct LatencyStatistics: Codable {
    let stageName: String
    let count: Int
    let meanMs: Double
    let stdMs: Double
    let minMs: Double
    let maxMs: Double
}

// MARK: - LatencyTracker

/// Lightweight stopwatch that records per-stage wall-clock durations using
/// `CACurrentMediaTime()` (Mach absolute time, nanosecond resolution).
///
/// Thread-safety: **not** thread-safe.  Use from one queue at a time.
final class LatencyTracker {

    /// Ordered list of stage names (insertion order).
    private var orderedStages: [String] = []

    /// Start timestamps (seconds).
    private var starts: [String: CFTimeInterval] = [:]

    /// Completed durations (milliseconds).
    private var durations: [String: Double] = [:]

    // MARK: API

    /// Mark the beginning of a stage.
    func start(_ stage: String) {
        if starts[stage] == nil && durations[stage] == nil {
            orderedStages.append(stage)
        }
        starts[stage] = CACurrentMediaTime()
    }

    /// Mark the end of a stage.  Duration is stored in milliseconds.
    func stop(_ stage: String) {
        guard let t0 = starts[stage] else { return }
        durations[stage] = (CACurrentMediaTime() - t0) * 1_000
        starts.removeValue(forKey: stage)
    }

    /// Build the pipeline result (ordered stages + total).
    func pipelineResult() -> PipelineLatency {
        let stages = orderedStages.compactMap { name -> StageLatency? in
            guard let ms = durations[name] else { return nil }
            return StageLatency(id: name, durationMs: ms)
        }
        let total = stages.reduce(0.0) { $0 + $1.durationMs }
        return PipelineLatency(stages: stages, totalMs: total)
    }

    /// Reset for the next image.
    func reset() {
        orderedStages.removeAll()
        starts.removeAll()
        durations.removeAll()
    }

    // MARK: - Aggregation (class-level helper)

    /// Compute per-stage mean/std/min/max over many pipeline runs.
    static func aggregate(_ runs: [PipelineLatency]) -> [LatencyStatistics] {
        guard !runs.isEmpty else { return [] }

        // Collect all unique stage names (preserving order).
        var seenStages: [String] = []
        var seenSet = Set<String>()
        for run in runs {
            for s in run.stages where !seenSet.contains(s.id) {
                seenStages.append(s.id)
                seenSet.insert(s.id)
            }
        }
        // Also add "total"
        seenStages.append("total_e2e")

        var result: [LatencyStatistics] = []

        for stageName in seenStages {
            let values: [Double]
            if stageName == "total_e2e" {
                values = runs.map(\.totalMs)
            } else {
                values = runs.compactMap { run in
                    run.stages.first(where: { $0.id == stageName })?.durationMs
                }
            }

            guard !values.isEmpty else { continue }

            let n = Double(values.count)
            let mean = values.reduce(0, +) / n
            let variance = values.reduce(0.0) { $0 + ($1 - mean) * ($1 - mean) } / max(n - 1, 1)
            let std = variance.squareRoot()

            result.append(LatencyStatistics(
                stageName: stageName,
                count: values.count,
                meanMs: mean,
                stdMs: std,
                minMs: values.min() ?? 0,
                maxMs: values.max() ?? 0
            ))
        }

        return result
    }

    /// Encode aggregated stats as JSON Data (for export / thesis log extraction).
    static func exportJSON(_ stats: [LatencyStatistics]) -> Data? {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return try? encoder.encode(stats)
    }
}
