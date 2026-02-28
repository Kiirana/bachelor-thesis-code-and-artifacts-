//
//  DetectionViewModel.swift
//  CurbDetectorApp
//
//  ViewModel handling texture classification + EV detection for single-image
//  analysis. Includes per-stage latency measurement via LatencyTracker
//  (CACurrentMediaTime) for thesis evaluation.
//

import SwiftUI
import Foundation
import CoreML
import Vision
import UIKit
import QuartzCore   // CACurrentMediaTime()

// MARK: - Output structs

struct TextureResult {
    let className: String
    let confidence: Float
    let allProbabilities: [String: Float]  // guaranteed to be in [0,1] and sum to ~1
}

struct Detection {
    let box: CGRect              // image coordinates (top-left origin, points)
    let confidence: Float
    let label: String
}

// MARK: - ViewModel

final class DetectionViewModel: ObservableObject {
    @Published var selectedImage: UIImage?
    @Published var annotatedImage: UIImage?

    @Published var groundCropImage: UIImage?
    @Published var groundCropBox: CGRect?

    @Published var textureResult: TextureResult?
    @Published var detectionResults: [Detection]?

    @Published var carDetection: Detection?

    @Published var isProcessing = false
    @Published var errorMessage: String?
    @Published var processingTime: Double = 0

    /// Per-stage latency for the most recent inference (for UI display).
    @Published var lastPipelineLatency: PipelineLatency?

    // Models
    private var textureModel: VNCoreMLModel?
    private var detectorModel: VNCoreMLModel?
    private var objectDetectorModel: VNCoreMLModel?

    // Fixed label order (must match model)
    private let textureClasses = ["asphalt", "cobblestone", "gravel", "sand"]

    // MARK: - Tuning

    private struct TextureTuning {
        /// Minimum top-1 probability to emit a label; below this → "unknown".
        var minTop1Confidence: Float = 0.50

        /// Per-class multiplicative weights (< 1 penalizes, > 1 boosts).
        var classBoost: [String: Float] = [
            "asphalt":     1,    // fully suppress asphalt (over-confident)
            "cobblestone": 1.0,
            "gravel":      1.0,
            "sand":        1.0,
        ]
    }

    private let textureTuning = TextureTuning()

    init() {
        loadModels()
    }

    private func loadModels() {
        textureModel        = loadVNModel(named: "MobileNetV3Texture")
        detectorModel       = loadVNModel(named: "YOLOv12EVCharger")
        objectDetectorModel = loadVNModel(named: "yolo12n")

        if textureModel != nil { print("✓ Texture model ready") }
        if detectorModel != nil { print("✓ EV detector ready") }
        if objectDetectorModel != nil { print("✓ YOLOv12 object detector ready") }
    }

    private func loadVNModel(named name: String) -> VNCoreMLModel? {
        for ext in ["mlmodelc", "mlpackage", "mlmodel"] {
            if let url = Bundle.main.url(forResource: name, withExtension: ext),
               let mlModel = try? MLModel(contentsOf: url),
               let vnModel = try? VNCoreMLModel(for: mlModel) {
                return vnModel
            }
        }
        print("Model \(name) not found in bundle")
        return nil
    }

    // MARK: - Main analysis entry point

    func analyzeImage() {
        guard let input = selectedImage else { return }

        isProcessing = true
        errorMessage = nil
        processingTime = 0
        lastPipelineLatency = nil

        textureResult = nil
        detectionResults = nil
        annotatedImage = nil
        groundCropImage = nil
        groundCropBox = nil
        carDetection = nil

        let startTime = CACurrentMediaTime()

        DispatchQueue.global(qos: .userInitiated).async { [weak self] in
            guard let self else { return }

            let tracker = LatencyTracker()
            let image = input.normalizedOrientation()
            let imgSize = image.size

            // --- Stage 1: Car detection (YOLOv12) ---
            tracker.start("yolo_car_detection")
            let car = self.detectCar(image: image)
            tracker.stop("yolo_car_detection")

            DispatchQueue.main.async { self.carDetection = car }

            // --- Stage 2: ROI extraction ---
            tracker.start("roi_extraction")
            let crop = self.extractGroundCrop(image: image, imageSize: imgSize, car: car)
            tracker.stop("roi_extraction")

            DispatchQueue.main.async { self.groundCropImage = crop }

            // --- Stage 3: Texture classification ---
            tracker.start("mobilenet_classification")
            if let crop {
                self.runTextureClassification(image: crop)
            } else {
                DispatchQueue.main.async {
                    self.errorMessage = "Could not create ground crop for texture analysis."
                }
            }
            tracker.stop("mobilenet_classification")

            // --- Stage 4: EV detection ---
            tracker.start("yolo_ev_detection")
            self.runEVDetection(image: image)
            tracker.stop("yolo_ev_detection")

            let pipelineResult = tracker.pipelineResult()
            let totalTime = (CACurrentMediaTime() - startTime) * 1_000

            DispatchQueue.main.async {
                self.isProcessing = false
                self.processingTime = totalTime / 1_000
                self.lastPipelineLatency = pipelineResult

                print("\n--- Pipeline Latency ---")
                for s in pipelineResult.stages {
                    print(String(format: "  %-28s  %.2f ms", s.id, s.durationMs))
                }
                print(String(format: "  %-28s  %.2f ms", "TOTAL (sum of stages)", pipelineResult.totalMs))
                print(String(format: "  %-28s  %.2f ms", "TOTAL (wall clock)", totalTime))
                print("------------------------\n")
            }
        }
    }

    // MARK: - Texture classification (robust probability handling)

    private func runTextureClassification(image: UIImage) {
        guard let model = textureModel else {
            DispatchQueue.main.async {
                self.errorMessage = "Texture model not loaded. Add MobileNetV3Texture to the project."
            }
            return
        }
        guard let ciImage = CIImage(image: image) else { return }

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self else { return }

            if let error {
                DispatchQueue.main.async {
                    self.errorMessage = "Texture classification failed: \(error.localizedDescription)"
                }
                return
            }

            // Case A: VNClassificationObservation (top-N softmax output)
            if let results = request.results as? [VNClassificationObservation], !results.isEmpty {
                var probs = self.emptyTextureProbDict()
                for r in results {
                    let key = self.normalizeTextureLabel(r.identifier)
                    if probs[key] != nil { probs[key] = Float(r.confidence) }
                }
                probs = self.sanitizeToProbabilities(probs)
                let tuned = self.tuneTextureProbabilities(probs)
                let (label, conf) = self.pickFinalTextureLabel(from: tuned)
                DispatchQueue.main.async {
                    self.textureResult = TextureResult(className: label, confidence: conf, allProbabilities: tuned)
                }
                return
            }

            // Case B: MLMultiArray logits/probs
            if let fv = request.results?.first as? VNCoreMLFeatureValueObservation,
               let arr = fv.featureValue.multiArrayValue {

                if arr.count != self.textureClasses.count {
                    DispatchQueue.main.async {
                        self.errorMessage = "Texture output (\(arr.count)) != labels (\(self.textureClasses.count))."
                    }
                    return
                }

                var raw: [Float] = []
                raw.reserveCapacity(arr.count)
                for i in 0..<arr.count { raw.append(arr[i].floatValue) }

                let probsArr: [Float] = self.looksLikeProbabilities(raw) ? self.renormalize(raw) : self.softmax(raw)

                var probs = self.emptyTextureProbDict()
                for i in 0..<self.textureClasses.count {
                    probs[self.textureClasses[i]] = probsArr[i]
                }
                probs = self.sanitizeToProbabilities(probs)
                let tuned = self.tuneTextureProbabilities(probs)
                let (label, conf) = self.pickFinalTextureLabel(from: tuned)
                DispatchQueue.main.async {
                    self.textureResult = TextureResult(className: label, confidence: conf, allProbabilities: tuned)
                }
                return
            }

            DispatchQueue.main.async {
                self.errorMessage = "Texture model returned an unsupported output type."
            }
        }

        request.imageCropAndScaleOption = .centerCrop
        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: image.cgImageOrientation, options: [:])
        try? handler.perform([request])
    }

    private func emptyTextureProbDict() -> [String: Float] {
        Dictionary(uniqueKeysWithValues: textureClasses.map { ($0, Float(0)) })
    }

    private func normalizeTextureLabel(_ s: String) -> String {
        s.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
    }

    // MARK: - Probability sanitization

    private func sanitizeToProbabilities(_ dict: [String: Float]) -> [String: Float] {
        let keys = textureClasses
        let vals = keys.map { dict[$0, default: 0] }
        if vals.contains(where: { $0 < 0 || $0 > 1.2 }) {
            let probsArr = softmax(vals)
            return Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, probsArr[$0]) })
        }
        let clamped = vals.map { max(0, min(1, $0)) }
        let s = clamped.reduce(0, +)
        if s <= 1e-12 {
            let u = 1.0 / Float(max(keys.count, 1))
            return Dictionary(uniqueKeysWithValues: keys.map { ($0, u) })
        }
        return Dictionary(uniqueKeysWithValues: keys.enumerated().map { ($1, clamped[$0] / s) })
    }

    // MARK: - Tuning pipeline

    private func tuneTextureProbabilities(_ probs: [String: Float]) -> [String: Float] {
        return applyClassBoosts(probs, boosts: textureTuning.classBoost)
    }

    private func pickFinalTextureLabel(from probs: [String: Float]) -> (String, Float) {
        let sorted = probs.sorted { $0.value > $1.value }
        guard let best = sorted.first else { return ("unknown", 0) }
        if best.value < textureTuning.minTop1Confidence {
            return ("unknown", best.value)
        }
        return (best.key, best.value)
    }

    // MARK: - Helper math

    private func applyClassBoosts(_ probs: [String: Float], boosts: [String: Float]) -> [String: Float] {
        var out: [String: Float] = [:]
        var sum: Float = 0
        for (k, p) in probs {
            let v = p * boosts[k, default: 1.0]
            out[k] = v; sum += v
        }
        guard sum > 0 else { return probs }
        for (k, v) in out { out[k] = v / sum }
        return out
    }

    private func looksLikeProbabilities(_ raw: [Float]) -> Bool {
        if raw.contains(where: { $0 < -1e-3 || $0 > 1.2 }) { return false }
        let s = raw.reduce(0, +)
        return s > 0.8 && s < 1.2
    }

    private func renormalize(_ x: [Float]) -> [Float] {
        let s = x.reduce(0, +); guard s > 0 else { return x }
        return x.map { $0 / s }
    }

    private func softmax(_ x: [Float]) -> [Float] {
        let m = x.max() ?? 0
        let exps = x.map { exp($0 - m) }
        let sum = exps.reduce(0, +)
        guard sum > 0 else { return Array(repeating: 0, count: x.count) }
        return exps.map { $0 / sum }
    }

    // MARK: - EV detection

    private func runEVDetection(image: UIImage) {
        guard let model = detectorModel else {
            DispatchQueue.main.async {
                self.errorMessage = "EV detector not loaded. Add YOLOv12EVCharger to the project."
            }
            return
        }
        guard let ciImage = CIImage(image: image) else { return }

        let request = VNCoreMLRequest(model: model) { [weak self] request, error in
            guard let self else { return }
            if let error {
                DispatchQueue.main.async { self.errorMessage = "EV detection failed: \(error.localizedDescription)" }
                return
            }
            let results = (request.results as? [VNRecognizedObjectObservation]) ?? []
            var detections: [Detection] = []
            detections.reserveCapacity(results.count)
            for obs in results {
                let box = self.convertBoundingBox(obs.boundingBox, imageSize: image.size)
                let label = obs.labels.first?.identifier ?? "ev_charger"
                detections.append(Detection(box: box, confidence: Float(obs.confidence), label: label))
            }
            let annotated = self.drawBoundingBoxes(on: image, detections: detections)
            DispatchQueue.main.async {
                self.detectionResults = detections
                self.annotatedImage = annotated
            }
        }
        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: image.cgImageOrientation, options: [:])
        try? handler.perform([request])
    }

    private func drawBoundingBoxes(on image: UIImage, detections: [Detection]) -> UIImage {
        let renderer = UIGraphicsImageRenderer(size: image.size)
        return renderer.image { context in
            image.draw(at: .zero)
            let ctx = context.cgContext
            for (index, detection) in detections.enumerated() {
                let color: UIColor = detection.confidence > 0.8 ? .green :
                                     detection.confidence > 0.5 ? .orange : .red
                ctx.setStrokeColor(color.cgColor)
                ctx.setLineWidth(3.0)
                ctx.stroke(detection.box)

                let label = "EV #\(index + 1): \(Int(detection.confidence * 100))%"
                let attrs: [NSAttributedString.Key: Any] = [
                    .font: UIFont.boldSystemFont(ofSize: 16),
                    .foregroundColor: UIColor.white
                ]
                let textSize = label.size(withAttributes: attrs)
                let textRect = CGRect(
                    x: detection.box.minX,
                    y: max(detection.box.minY - textSize.height - 8, 0),
                    width: textSize.width + 12,
                    height: textSize.height + 6
                )
                ctx.setFillColor(color.withAlphaComponent(0.8).cgColor)
                ctx.fill(textRect)
                label.draw(at: CGPoint(x: textRect.minX + 6, y: textRect.minY + 3), withAttributes: attrs)
            }
        }
    }

    // MARK: - Car detection (YOLOv12) + crop logic

    private func detectCar(image: UIImage) -> Detection? {
        guard let model = objectDetectorModel,
              let ciImage = CIImage(image: image) else { return nil }

        var best: Detection?
        let vehicleLabels: Set<String> = ["car", "truck", "bus", "motorcycle"]

        let request = VNCoreMLRequest(model: model) { [weak self] request, _ in
            guard let self else { return }
            guard let results = request.results as? [VNRecognizedObjectObservation] else { return }
            for obs in results {
                guard let top = obs.labels.first else { continue }
                let label = top.identifier.lowercased()
                guard vehicleLabels.contains(label), Float(top.confidence) >= 0.30 else { continue }
                let box = self.convertBoundingBox(obs.boundingBox, imageSize: image.size)
                let det = Detection(box: box, confidence: Float(top.confidence), label: label)
                if best == nil || det.confidence > best!.confidence { best = det }
            }
        }

        request.imageCropAndScaleOption = .scaleFit
        let handler = VNImageRequestHandler(ciImage: ciImage, orientation: image.cgImageOrientation, options: [:])
        try? handler.perform([request])
        return best
    }

    private func extractGroundCrop(image: UIImage, imageSize: CGSize, car: Detection?) -> UIImage? {
        let rect: CGRect = car.map { groundRectBelow(carBox: $0.box, imageSize: imageSize) }
                        ?? defaultGroundRect(imageSize: imageSize)
        DispatchQueue.main.async { self.groundCropBox = rect }
        return cropImage(image, to: rect)
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

    // MARK: - Geometry / image utils

    private func convertBoundingBox(_ bb: CGRect, imageSize: CGSize) -> CGRect {
        let w = bb.width * imageSize.width
        let h = bb.height * imageSize.height
        let x = bb.minX * imageSize.width
        let y = (1 - bb.maxY) * imageSize.height
        return CGRect(x: x, y: y, width: w, height: h)
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
}

// MARK: - UIImage helpers

extension UIImage {
    func normalizedOrientation() -> UIImage {
        if imageOrientation == .up { return self }
        UIGraphicsBeginImageContextWithOptions(size, false, scale)
        defer { UIGraphicsEndImageContext() }
        draw(in: CGRect(origin: .zero, size: size))
        return UIGraphicsGetImageFromCurrentImageContext() ?? self
    }

    var cgImageOrientation: CGImagePropertyOrientation {
        switch imageOrientation {
        case .up:            return .up
        case .down:          return .down
        case .left:          return .left
        case .right:         return .right
        case .upMirrored:    return .upMirrored
        case .downMirrored:  return .downMirrored
        case .leftMirrored:  return .leftMirrored
        case .rightMirrored: return .rightMirrored
        @unknown default:    return .up
        }
    }
}

