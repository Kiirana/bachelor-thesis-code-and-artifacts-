//
//  BatchEvaluationViewModel.swift
//  CurbDetectorApp
//
//  Runs the full pipeline (YOLO car detection → ROI crop → MobileNetV3 texture
//  classification → YOLO EV-charger detection) over a dataset of images and
//  collects per-stage latency using CACurrentMediaTime().
//
//  Thesis usage:
//    1. Add a folder "EvalDataset" to the app bundle (Target → Build Phases →
//       Copy Bundle Resources).  Put ≤ 100 JPEG/PNG images inside.
//    2. Tap "Run Batch Evaluation" in the UI.
//    3. Results are printed to the Xcode console AND saved as JSON in the
//       app's Documents directory for extraction via Xcode / Files app.
//
//  The exported JSON matches the thesis table schema:
//    { stageName, count, meanMs, stdMs, minMs, maxMs }
//
//  Reference:
//    Apple – CACurrentMediaTime()
//    https://developer.apple.com/documentation/quartzcore/1395996-cacurrentmediatime
//

import SwiftUI
import CoreML
import Vision
import UIKit
import QuartzCore

// MARK: - Per-image result

struct BatchImageResult: Identifiable {
    let id: Int                     // image index (0-based)
    let filename: String
    let groundTruth: String?        // derived from filename prefix (e.g. "asphalt" from "asphalt_01.jpg")
    let latency: PipelineLatency
    let evDetections: Int           // number of EV chargers found
    let textureClass: String?       // top-1 texture label (nil if crop failed)
    let textureConfidence: Float?
    let carDetected: Bool

    /// true when model prediction matches the ground-truth label
    var isCorrect: Bool? {
        guard let gt = groundTruth, let pred = textureClass else { return nil }
        return gt == pred
    }
}

// MARK: - ViewModel

@MainActor
final class BatchEvaluationViewModel: ObservableObject {

    // Published state for the UI
    @Published var isRunning = false
    @Published var progress: Double = 0          // 0 … 1
    @Published var currentImage: String = ""
    @Published var imageResults: [BatchImageResult] = []
    @Published var aggregatedStats: [LatencyStatistics] = []
    @Published var exportedFilePath: String?
    @Published var errorMessage: String?

    // Models (same as DetectionViewModel)
    private var textureModel: VNCoreMLModel?
    private var detectorModel: VNCoreMLModel?
    private var objectDetectorModel: VNCoreMLModel?

    private let textureClasses = ["asphalt", "cobblestone", "gravel", "sand"]

    // MARK: - Tuning (mirrors DetectionViewModel)

    private struct TextureTuning {
        /// Minimum top-1 probability to emit a label; below this → "unknown".
        var minTop1Confidence: Float = 0.45

        /// Per-class multiplicative weights (< 1 penalizes, > 1 boosts).
        var classBoost: [String: Float] = [
            "asphalt":     1.0,    // fully suppress asphalt (over-confident)
            "cobblestone": 1.0,
            "gravel":      1.0,
            "sand":        1.0,
        ]
    }

    private let textureTuning = TextureTuning()

    /// Maps bare filename (e.g. "IMG_4531.HEIC") → ground-truth label.
    /// Loaded from a JSON manifest (100BatchrealTest.json or labels.json) in the bundle.
    private var groundTruthManifest: [String: String] = [:]

    init() {
        loadModels()
        loadManifest()
    }

    // MARK: - Manifest loading

    /// Searches the bundle for a JSON manifest file containing [{filename, groundTruth}] entries.
    /// Accepts both the 100BatchrealTest.json format (full paths) and a simple {"IMG_4531.HEIC": "asphalt"} dict.
    private func loadManifest() {
        let candidateNames = ["100BatchrealTest", "labels", "manifest", "groundTruth"]
        let extensions = ["json"]

        for name in candidateNames {
            for ext in extensions {
                guard let url = Bundle.main.url(forResource: name, withExtension: ext),
                      let data = try? Data(contentsOf: url) else { continue }

                // Try array format: [{"filename": "...", "groundTruth": "asphalt", ...}]
                if let array = try? JSONSerialization.jsonObject(with: data) as? [[String: String]] {
                    var map: [String: String] = [:]
                    for entry in array {
                        guard let rawPath = entry["filename"],
                              let gt = entry["groundTruth"] else { continue }
                        // Use only the last path component so full macOS paths work too
                        let key = URL(fileURLWithPath: rawPath).lastPathComponent
                        map[key] = gt
                    }
                    if !map.isEmpty {
                        groundTruthManifest = map
                        print("✅ Loaded ground-truth manifest '\(name).\(ext)': \(map.count) entries")
                        return
                    }
                }

                // Try dict format: {"IMG_4531.HEIC": "asphalt", ...}
                if let dict = try? JSONSerialization.jsonObject(with: data) as? [String: String] {
                    if !dict.isEmpty {
                        groundTruthManifest = dict
                        print("✅ Loaded ground-truth manifest '\(name).\(ext)': \(dict.count) entries")
                        return
                    }
                }
            }
        }
        print("ℹ️ No JSON manifest found — ground truth will be parsed from filename prefix.")
    }

    // MARK: - Model loading

    private func loadModels() {
        textureModel        = loadVNModel(named: "MobileNetV3Texture")
        detectorModel       = loadVNModel(named: "YOLOv12EVCharger")
        objectDetectorModel = loadVNModel(named: "yolo12n")
    }

    private func loadVNModel(named name: String) -> VNCoreMLModel? {
        for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
            if let url = Bundle.main.url(forResource: name, withExtension: ext),
               let mlModel = try? MLModel(contentsOf: url),
               let vnModel = try? VNCoreMLModel(for: mlModel) {
                return vnModel
            }
        }
        print("⚠️ Model \(name) not found in bundle")
        return nil
    }

    // MARK: - Dataset discovery

    /// Returns image URLs from the "EvalDataset" bundle folder.
    private func discoverDataset() -> [URL] {
        let validExtensions: Set<String> = ["jpg", "jpeg", "png", "heic"]

        // Option 1: folder reference in bundle
        if let folderURL = Bundle.main.url(forResource: "EvalDataset", withExtension: nil) {
            let fm = FileManager.default
            if let contents = try? fm.contentsOfDirectory(at: folderURL,
                                                           includingPropertiesForKeys: nil) {
                let images = contents
                    .filter { validExtensions.contains($0.pathExtension.lowercased()) }
                    .sorted { $0.lastPathComponent < $1.lastPathComponent }
                return images
            }
        }

        // Option 2: flat files prefixed "eval_" in bundle root
        var results: [URL] = []
        for ext in validExtensions {
            if let urls = Bundle.main.urls(forResourcesWithExtension: ext, subdirectory: nil) {
                results.append(contentsOf: urls.filter {
                    $0.lastPathComponent.lowercased().hasPrefix("eval_")
                })
            }
        }
        return results.sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    // MARK: - Run batch evaluation

    func runBatchEvaluation() {
        guard !isRunning else { return }

        let images = discoverDataset()
        guard !images.isEmpty else {
            errorMessage = "No images found.  Add a folder \"EvalDataset\" to the app bundle (Copy Bundle Resources)."
            return
        }

        isRunning = true
        progress = 0
        imageResults = []
        aggregatedStats = []
        exportedFilePath = nil
        errorMessage = nil

        let total = images.count

        // Run on background queue, publish results back to main
        Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }

            var pipelineRuns: [PipelineLatency] = []
            var batchResults: [BatchImageResult] = []

            for (index, url) in images.enumerated() {
                let filename = url.lastPathComponent

                await MainActor.run {
                    self.currentImage = filename
                    self.progress = Double(index) / Double(total)
                }

                guard let data = try? Data(contentsOf: url),
                      let uiImage = UIImage(data: data)?.normalizedOrientation() else {
                    continue
                }

                let (result, latency) = await self.runPipeline(image: uiImage, index: index, filename: filename)
                pipelineRuns.append(latency)
                batchResults.append(result)
            }

            // Aggregate
            let stats = LatencyTracker.aggregate(pipelineRuns)

            // Export JSON
            let exportPath = await self.exportResults(stats: stats, perImage: batchResults)

            // Print thesis-formatted summary to console
            await self.printThesisSummary(stats: stats, perImage: batchResults)

            await MainActor.run {
                self.imageResults = batchResults
                self.aggregatedStats = stats
                self.exportedFilePath = exportPath
                self.progress = 1.0
                self.currentImage = "Done"
                self.isRunning = false
            }
        }
    }

    // MARK: - Single-image pipeline (with timing)

    private func runPipeline(image: UIImage, index: Int, filename: String) async -> (BatchImageResult, PipelineLatency) {
        let tracker = LatencyTracker()
        let imgSize = image.size

        // --- Stage 1: YOLO12n car detection --- [DISABLED FOR TESTING: always use fallback]
        tracker.start("yolo_car_detection")
       let carDetection = detectCar(image: image)
       // let carDetection: Detection? = nil
        tracker.stop("yolo_car_detection")

        // --- Stage 2: ROI extraction ---
       tracker.start("roi_extraction")
        let groundCrop: UIImage?
        if let car = carDetection {
          let groundRect = groundRectBelow(carBox: car.box, imageSize: imgSize)
    groundCrop = cropImage(image, to: groundRect)
        } else {
            let fallback = defaultGroundRect(imageSize: imgSize)
            groundCrop = cropImage(image, to: fallback)
   
      }
        tracker.stop("roi_extraction")

        // --- Stage 3: MobileNetV3 texture classification ---
        tracker.start("mobilenet_classification")
        var texClass: String? = nil
        var texConf: Float? = nil
        if let crop = groundCrop {
            let (label, conf) = classifyTexture(image: crop)
            texClass = label
            texConf = conf
        }
        tracker.stop("mobilenet_classification")

        // --- Stage 4: YOLO EV-charger detection ---
        tracker.start("yolo_ev_detection")
        let evCount = detectEVChargers(image: image)
        tracker.stop("yolo_ev_detection")

        // Derive ground truth: 1) JSON manifest, 2) filename prefix fallback
        let knownClasses: Set<String> = ["asphalt", "cobblestone", "gravel", "sand"]
        let groundTruth: String?
        if let gt = groundTruthManifest[filename], knownClasses.contains(gt) {
            groundTruth = gt
        } else {
            // Fallback: "asphalt_01.jpg" → "asphalt"
            let prefix = filename.components(separatedBy: "_").first?.lowercased() ?? ""
            groundTruth = knownClasses.contains(prefix) ? prefix : nil
        }

        let latency = tracker.pipelineResult()
        let result = BatchImageResult(
            id: index,
            filename: filename,
            groundTruth: groundTruth,
            latency: latency,
            evDetections: evCount,
            textureClass: texClass,
            textureConfidence: texConf,
            carDetected: carDetection != nil
        )
        return (result, latency)
    }

    // MARK: - Inference helpers (synchronous, for batch use)

    private func detectCar(image: UIImage) -> Detection? {
        guard let model = objectDetectorModel,
              let ciImage = CIImage(image: image) else { return nil }

        var bestCar: Detection?
        let vehicleLabels: Set<String> = ["car", "truck", "bus", "motorcycle"]

        let request = VNCoreMLRequest(model: model) { request, _ in
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            for obs in results {
                guard let top = obs.labels.first else { continue }
                let label = top.identifier.lowercased()
                guard vehicleLabels.contains(label), Float(top.confidence) >= 0.30 else { continue }
                let box = self.convertBoundingBox(obs.boundingBox, imageSize: image.size)
                let det = Detection(box: box, confidence: Float(top.confidence), label: label)
                if bestCar == nil || det.confidence > bestCar!.confidence {
                    bestCar = det
                }
            }
        }
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(ciImage: ciImage,
                                            orientation: image.cgImageOrientation,
                                            options: [:])
        try? handler.perform([request])
        return bestCar
    }

    private func classifyTexture(image: UIImage) -> (String, Float) {
        guard let model = textureModel,
              let ciImage = CIImage(image: image) else { return ("unknown", 0) }

        var bestLabel = "unknown"
        var bestConf: Float = 0

        let request = VNCoreMLRequest(model: model) { [self] request, _ in
            if let results = request.results as? [VNClassificationObservation], !results.isEmpty {
                var probs: [String: Float] = Dictionary(uniqueKeysWithValues: textureClasses.map { ($0, Float(0)) })
                for r in results {
                    let key = r.identifier.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    if probs[key] != nil { probs[key] = Float(r.confidence) }
                }
                probs = sanitizeToProbabilities(probs)
                let tuned = tuneTextureProbabilities(probs)
                let sorted = tuned.sorted { $0.value > $1.value }
                if let top = sorted.first, top.value >= textureTuning.minTop1Confidence {
                    bestLabel = top.key
                    bestConf = top.value
                }
            } else if let fv = request.results?.first as? VNCoreMLFeatureValueObservation,
                      let arr = fv.featureValue.multiArrayValue,
                      arr.count == textureClasses.count {
                var raw: [Float] = (0..<arr.count).map { arr[$0].floatValue }
                let probsArr: [Float] = looksLikeProbabilities(raw) ? renormalize(raw) : softmax(raw)
                var probs: [String: Float] = [:]
                for i in 0..<textureClasses.count { probs[textureClasses[i]] = probsArr[i] }
                probs = sanitizeToProbabilities(probs)
                let tuned = tuneTextureProbabilities(probs)
                let sorted = tuned.sorted { $0.value > $1.value }
                if let top = sorted.first, top.value >= textureTuning.minTop1Confidence {
                    bestLabel = top.key
                    bestConf = top.value
                }
            }
        }
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(ciImage: ciImage,
                                            orientation: image.cgImageOrientation,
                                            options: [:])
        try? handler.perform([request])
        return (bestLabel, bestConf)
    }

    // MARK: - Probability helpers (mirrors DetectionViewModel)

    private func sanitizeToProbabilities(_ dict: [String: Float]) -> [String: Float] {
        let keys = textureClasses
        let vals = keys.map { dict[$0, default: 0] }
        if vals.contains(where: { $0 < 0 || $0 > 1.2 }) {
            let p = softmax(vals)
            return Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, p[$0]) })
        }
        let clamped = vals.map { max(0, min(1, $0)) }
        let s = clamped.reduce(0, +)
        if s <= 1e-12 {
            let u = 1.0 / Float(max(keys.count, 1))
            return Dictionary(uniqueKeysWithValues: keys.map { ($0, u) })
        }
        return Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, clamped[$0] / s) })
    }

    private func tuneTextureProbabilities(_ probs: [String: Float]) -> [String: Float] {
        return applyClassBoosts(probs, boosts: textureTuning.classBoost)
    }

    private func applyClassBoosts(_ probs: [String: Float], boosts: [String: Float]) -> [String: Float] {
        var out: [String: Float] = [:]; var sum: Float = 0
        for (k, p) in probs { let v = p * boosts[k, default: 1.0]; out[k] = v; sum += v }
        guard sum > 0 else { return probs }
        for (k, v) in out { out[k] = v / sum }
        return out
    }

    private func looksLikeProbabilities(_ raw: [Float]) -> Bool {
        if raw.contains(where: { $0 < -1e-3 || $0 > 1.2 }) { return false }
        let s = raw.reduce(0, +); return s > 0.8 && s < 1.2
    }

    private func renormalize(_ x: [Float]) -> [Float] {
        let s = x.reduce(0, +); guard s > 0 else { return x }; return x.map { $0 / s }
    }

    private func softmax(_ x: [Float]) -> [Float] {
        let m = x.max() ?? 0
        let exps = x.map { exp($0 - m) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return Array(repeating: 0, count: x.count) }
        return exps.map { $0 / sum }
    }

    private func detectEVChargers(image: UIImage) -> Int {
        guard let model = detectorModel,
              let ciImage = CIImage(image: image) else { return 0 }

        var count = 0
        let request = VNCoreMLRequest(model: model) { request, _ in
            count = (request.results as? [VNRecognizedObjectObservation])?.count ?? 0
        }
        request.imageCropAndScaleOption = .scaleFit

        let handler = VNImageRequestHandler(ciImage: ciImage,
                                            orientation: image.cgImageOrientation,
                                            options: [:])
        try? handler.perform([request])
        return count
    }

    // MARK: - Geometry (same logic as DetectionViewModel)

    private func convertBoundingBox(_ bb: CGRect, imageSize: CGSize) -> CGRect {
        let w = bb.width * imageSize.width
        let h = bb.height * imageSize.height
        let x = bb.minX * imageSize.width
        let y = (1 - bb.maxY) * imageSize.height
        return CGRect(x: x, y: y, width: w, height: h)
    }

    private func groundRectBelow(carBox: CGRect, imageSize: CGSize) -> CGRect {
        let imgW = imageSize.width; let imgH = imageSize.height
        let gap = max(10, imgH * 0.015)
        let yTop = min(imgH, carBox.maxY + gap)
        if yTop >= imgH - 40 { return defaultGroundRect(imageSize: imageSize) }
        let cropW = min(max(220, carBox.width * 0.9), imgW)
        var xLeft = carBox.midX - cropW / 2
        xLeft = min(max(xLeft, 0), imgW - cropW)
        return CGRect(x: xLeft, y: yTop, width: cropW, height: imgH - yTop)
    }

    private func defaultGroundRect(imageSize: CGSize) -> CGRect {
        let h = max(180, imageSize.height * 0.25)
        return CGRect(x: 0, y: imageSize.height - h, width: imageSize.width, height: h)
    }
    
    

    private func cropImage(_ image: UIImage, to rect: CGRect) -> UIImage? {
        guard let cg = image.cgImage else { return nil }
        let s = image.scale
        let scaled = CGRect(x: rect.origin.x * s, y: rect.origin.y * s,
                            width: rect.size.width * s, height: rect.size.height * s).integral
        guard scaled.width > 2, scaled.height > 2,
              let cropped = cg.cropping(to: scaled) else { return nil }
        return UIImage(cgImage: cropped, scale: image.scale, orientation: .up)
    }

    // MARK: - Export

    private func exportResults(stats: [LatencyStatistics], perImage: [BatchImageResult]) async -> String? {
        let docs = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask).first
        guard let dir = docs else { return nil }

        // 1) Aggregated stats JSON (for thesis table)
        let statsFile = dir.appendingPathComponent("latency_stats.json")
        if let data = LatencyTracker.exportJSON(stats) {
            try? data.write(to: statsFile)
        }

        // 2) Per-image CSV (for histograms / boxplots)
        let csvFile = dir.appendingPathComponent("latency_per_image.csv")
        var csv = "index,filename,ground_truth,texture_class,correct,texture_conf,car_detected,ev_detections,"
        // dynamic stage columns
        let stageNames = stats.filter { $0.stageName != "total_e2e" }.map(\.stageName)
        csv += stageNames.joined(separator: ",") + ",total_ms\n"

        for r in perImage {
            let correctStr: String
            if let c = r.isCorrect { correctStr = c ? "1" : "0" } else { correctStr = "" }
            var line = "\(r.id),\(r.filename),\(r.groundTruth ?? ""),\(r.textureClass ?? "none"),\(correctStr),"
            line += String(format: "%.4f", r.textureConfidence ?? 0)
            line += ",\(r.carDetected),\(r.evDetections)"
            for stage in stageNames {
                let ms = r.latency.stages.first(where: { $0.id == stage })?.durationMs ?? 0
                line += String(format: ",%.2f", ms)
            }
            line += String(format: ",%.2f", r.latency.totalMs)
            csv += line + "\n"
        }
        try? csv.write(to: csvFile, atomically: true, encoding: .utf8)

        return statsFile.path
    }

    // MARK: - Console summary

    private func printThesisSummary(stats: [LatencyStatistics], perImage: [BatchImageResult]) async {
        let imageCount = perImage.count
        print("\n" + String(repeating: "=", count: 70))
        print("BATCH EVALUATION COMPLETE — \(imageCount) images")
        print(String(repeating: "=", count: 70))
        print(String(format: "%-28@ %8@ %8@ %8@ %8@", "Stage" as NSString, "Mean(ms)" as NSString, "Std(ms)" as NSString, "Min(ms)" as NSString, "Max(ms)" as NSString))
        print(String(repeating: "-", count: 70))
        for s in stats {
            print(String(format: "%-28@ %8.2f %8.2f %8.2f %8.2f",
                         s.stageName as NSString, s.meanMs, s.stdMs, s.minMs, s.maxMs))
        }

        // Per-image correctness table
        print("\n" + String(repeating: "-", count: 70))
        print(String(format: "%-6@ %-30@ %-14@ %-14@ %@",
                     "#" as NSString, "File" as NSString,
                     "Truth" as NSString, "Predicted" as NSString, "OK?" as NSString))
        print(String(repeating: "-", count: 70))
        for r in perImage {
            let truth     = r.groundTruth ?? "?"
            let predicted = r.textureClass ?? "none"
            let ok: String
            if let c = r.isCorrect { ok = c ? "✓" : "✗" } else { ok = "-" }
            // Truncate filename to 30 chars for readability
            let name = r.filename.count > 30
                ? "..." + r.filename.suffix(27)
                : r.filename
            print(String(format: "%-6d %-30@ %-14@ %-14@ %@",
                         r.id + 1,
                         name as NSString,
                         truth as NSString,
                         predicted as NSString,
                         ok as NSString))
        }

        // Texture accuracy summary
        let labelled = perImage.filter { $0.groundTruth != nil && $0.textureClass != nil }
        if !labelled.isEmpty {
            let correct = labelled.filter { $0.isCorrect == true }.count
            let acc = Double(correct) / Double(labelled.count) * 100
            print(String(repeating: "-", count: 70))
            print(String(format: "Texture accuracy: %d / %d  (%.1f%%)", correct, labelled.count, acc))
            // Per-class breakdown
            let classes = ["asphalt", "cobblestone", "gravel", "sand"]
            for cls in classes {
                let forCls = labelled.filter { $0.groundTruth == cls }
                let cCorrect = forCls.filter { $0.isCorrect == true }.count
                if !forCls.isEmpty {
                    print(String(format: "  %-14@ %d/%d  (%.0f%%)",
                                 cls as NSString, cCorrect, forCls.count,
                                 Double(cCorrect)/Double(forCls.count)*100))
                }
            }
        } else {
            print(String(repeating: "-", count: 70))
            print("No ground-truth labels found — ensure 100BatchrealTest.json is in Copy Bundle Resources.")
        }
        print(String(repeating: "=", count: 70))
        print("Results exported to Documents/latency_stats.json")
        print("Per-image CSV: Documents/latency_per_image.csv\n")
    }
}

