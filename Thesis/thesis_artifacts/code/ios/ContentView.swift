//
//  ContentView.swift
//  CurbDetectorApp
//
//  Combined Texture Classification + EV Charger Detection + Batch Evaluation.
//

import SwiftUI
import UIKit

struct ContentView: View {
    @StateObject private var viewModel = DetectionViewModel()
    @State private var showCamera = false
    @State private var showBoundingBoxes = true

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {

                    // MARK: - Image Display
                    if let image = viewModel.selectedImage {
                        VStack(spacing: 12) {
                            ZStack {
                                Image(uiImage: image)
                                    .resizable()
                                    .scaledToFit()
                                    .frame(maxHeight: 300)

                                if showBoundingBoxes, let annotated = viewModel.annotatedImage {
                                    Image(uiImage: annotated)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxHeight: 300)
                                }

                                if showBoundingBoxes {
                                    GeometryReader { geo in
                                        let container = geo.size
                                        let imgSize = image.size
                                        let scale = min(container.width / imgSize.width,
                                                        container.height / imgSize.height)
                                        let fittedW = imgSize.width * scale
                                        let fittedH = imgSize.height * scale
                                        let xOffset = (container.width - fittedW) / 2
                                        let yOffset = (container.height - fittedH) / 2

                                        if let car = viewModel.carDetection {
                                            Rectangle()
                                                .stroke(Color.green, lineWidth: 2)
                                                .frame(width: car.box.width * scale,
                                                       height: car.box.height * scale)
                                                .position(x: xOffset + car.box.midX * scale,
                                                           y: yOffset + car.box.midY * scale)
                                        }

                                        if let ground = viewModel.groundCropBox {
                                            Rectangle()
                                                .stroke(Color.blue, lineWidth: 2)
                                                .frame(width: ground.width * scale,
                                                       height: ground.height * scale)
                                                .position(x: xOffset + ground.midX * scale,
                                                           y: yOffset + ground.midY * scale)
                                        }
                                    }
                                    .allowsHitTesting(false)
                                }
                            }
                            .cornerRadius(12)
                            .shadow(radius: 5)

                            // Ground crop preview
                            if let groundCrop = viewModel.groundCropImage {
                                VStack(spacing: 4) {
                                    Text("Ground Crop (for texture analysis)")
                                        .font(.caption)
                                        .foregroundColor(.gray)
                                    Image(uiImage: groundCrop)
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxHeight: 120)
                                        .cornerRadius(8)
                                        .overlay(
                                            RoundedRectangle(cornerRadius: 8)
                                                .stroke(Color.blue, lineWidth: 2)
                                        )
                                }
                            }
                        }
                    } else {
                        RoundedRectangle(cornerRadius: 12)
                            .fill(Color.gray.opacity(0.2))
                            .frame(height: 400)
                            .overlay(
                                VStack {
                                    Image(systemName: "camera.fill")
                                        .font(.system(size: 60))
                                        .foregroundColor(.gray)
                                    Text("Take a photo to analyze")
                                        .foregroundColor(.gray)
                                        .padding(.top)
                                }
                            )
                    }

                    // MARK: - Buttons
                    Button {
                        showCamera = true
                    } label: {
                        HStack {
                            Image(systemName: "camera.fill")
                            Text("Take Photo")
                        }
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.blue)
                        .cornerRadius(12)
                    }

                    Button {
                        loadTestImage()
                    } label: {
                        HStack {
                            Image(systemName: "photo.fill")
                            Text("Load Test Image")
                        }
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.purple)
                        .cornerRadius(12)
                    }

                    if viewModel.selectedImage != nil {
                        Button {
                            viewModel.analyzeImage()
                        } label: {
                            HStack {
                                if viewModel.isProcessing {
                                    ProgressView()
                                        .progressViewStyle(CircularProgressViewStyle(tint: .white))
                                } else {
                                    Image(systemName: "bolt.fill")
                                    Text("Analyze Image")
                                }
                            }
                            .font(.headline)
                            .foregroundColor(.white)
                            .frame(maxWidth: .infinity)
                            .padding()
                            .background(viewModel.isProcessing ? Color.gray : Color.green)
                            .cornerRadius(12)
                        }
                        .disabled(viewModel.isProcessing)
                    }

                    // MARK: - Batch Evaluation Navigation
                    NavigationLink(destination: BatchEvaluationView()) {
                        HStack {
                            Image(systemName: "chart.bar.doc.horizontal")
                            Text("Batch Evaluation (100 images)")
                        }
                        .font(.headline)
                        .foregroundColor(.white)
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.indigo)
                        .cornerRadius(12)
                    }

                    // MARK: - Toggle boxes
                    if viewModel.detectionResults != nil || viewModel.carDetection != nil || viewModel.groundCropBox != nil {
                        Toggle(isOn: $showBoundingBoxes) {
                            HStack {
                                Image(systemName: showBoundingBoxes ? "eye.fill" : "eye.slash.fill")
                                Text("Show Bounding Boxes")
                            }
                        }
                        .padding()
                        .background(Color.gray.opacity(0.1))
                        .cornerRadius(12)
                    }

                    // MARK: - Per-stage Latency Display
                    if let latency = viewModel.lastPipelineLatency {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Pipeline Latency")
                                .font(.headline)
                                .foregroundColor(.indigo)

                            ForEach(latency.stages) { stage in
                                HStack {
                                    Text(stage.id)
                                        .font(.caption)
                                    Spacer()
                                    Text(String(format: "%.3f ms", stage.durationMs))
                                        .font(.system(.caption, design: .monospaced))
                                        .foregroundColor(.secondary)
                                }
                            }

                            Divider()

                            HStack {
                                Text("Total (sum)")
                                    .font(.caption).fontWeight(.semibold)
                                Spacer()
                                Text(String(format: "%.3f ms", latency.totalMs))
                                    .font(.system(.caption, design: .monospaced))
                                    .fontWeight(.semibold)
                            }
                        }
                        .padding()
                        .background(Color.indigo.opacity(0.08))
                        .cornerRadius(12)
                    }

                    // MARK: - Texture Result
                    if let texture = viewModel.textureResult {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("Ground Texture Analysis")
                                .font(.headline)
                                .foregroundColor(.blue)

                            HStack {
                                VStack(alignment: .leading) {
                                    Text("Classification:")
                                        .font(.subheadline).foregroundColor(.gray)
                                    Text(texture.className.capitalized)
                                        .font(.title2).fontWeight(.bold)
                                }
                                Spacer()
                                VStack(alignment: .trailing) {
                                    Text("Confidence:")
                                        .font(.subheadline).foregroundColor(.gray)
                                    Text(String(format: "%.1f%%", texture.confidence * 100))
                                        .font(.title2).fontWeight(.bold)
                                        .foregroundColor(confidenceColor(texture.confidence))
                                }
                            }

                            ForEach(texture.allProbabilities.sorted(by: { $0.value > $1.value }), id: \.key) { item in
                                HStack {
                                    Text(item.key.capitalized).font(.caption)
                                    Spacer()
                                    Text(String(format: "%.1f%%", item.value * 100))
                                        .font(.caption).foregroundColor(.gray)
                                }
                            }
                        }
                        .padding()
                        .background(Color.blue.opacity(0.1))
                        .cornerRadius(12)
                    }

                    // MARK: - EV Results
                    if let detectionResults = viewModel.detectionResults {
                        VStack(alignment: .leading, spacing: 12) {
                            Text("EV Charger Detection")
                                .font(.headline).foregroundColor(.green)

                            if detectionResults.isEmpty {
                                Text("No EV chargers detected")
                                    .foregroundColor(.gray).italic()
                            } else {
                                Text("\(detectionResults.count) charger(s) detected")
                                    .font(.subheadline).foregroundColor(.green)

                                ForEach(Array(detectionResults.enumerated()), id: \.offset) { index, detection in
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack {
                                            Text("Charger #\(index + 1)")
                                                .font(.subheadline).fontWeight(.semibold)
                                            Spacer()
                                            Text(String(format: "%.1f%%", detection.confidence * 100))
                                                .font(.caption)
                                                .padding(.horizontal, 8).padding(.vertical, 4)
                                                .background(confidenceColor(detection.confidence).opacity(0.2))
                                                .foregroundColor(confidenceColor(detection.confidence))
                                                .cornerRadius(8)
                                        }
                                        Text("Box: x=\(Int(detection.box.minX)), y=\(Int(detection.box.minY)), w=\(Int(detection.box.width))×h=\(Int(detection.box.height))")
                                            .font(.caption2).foregroundColor(.gray)
                                    }
                                    .padding(.vertical, 4)
                                    if index < detectionResults.count - 1 { Divider() }
                                }
                            }
                        }
                        .padding()
                        .background(Color.green.opacity(0.1))
                        .cornerRadius(12)
                    }

                    // MARK: - Car Detection
                    if let car = viewModel.carDetection {
                        VStack(alignment: .leading, spacing: 8) {
                            Text("Object Detection (YOLOv12)")
                                .font(.headline).foregroundColor(.orange)
                            HStack {
                                VStack(alignment: .leading) {
                                    Text("Detected: \(car.label.capitalized)")
                                        .font(.subheadline).fontWeight(.semibold)
                                    Text("Used to place the ground crop box")
                                        .font(.caption).foregroundColor(.gray)
                                }
                                Spacer()
                                Text(String(format: "%.1f%%", car.confidence * 100))
                                    .font(.subheadline)
                                    .padding(.horizontal, 10).padding(.vertical, 5)
                                    .background(Color.orange.opacity(0.2))
                                    .foregroundColor(.orange)
                                    .cornerRadius(8)
                            }
                        }
                        .padding()
                        .background(Color.orange.opacity(0.1))
                        .cornerRadius(12)
                    }

                    // MARK: - Processing time (legacy display)
                    if viewModel.processingTime > 0 {
                        Text("Processing time: \(String(format: "%.2f", viewModel.processingTime))s")
                            .font(.caption).foregroundColor(.gray)
                    }

                    // MARK: - Error
                    if let error = viewModel.errorMessage {
                        Text(error)
                            .foregroundColor(.red)
                            .font(.caption)
                            .padding()
                            .background(Color.red.opacity(0.1))
                            .cornerRadius(8)
                    }
                }
                .padding()
            }
            .navigationTitle("Ground & Charger Detector")
            .navigationBarTitleDisplayMode(.inline)
            .sheet(isPresented: $showCamera) {
                CameraView(image: $viewModel.selectedImage)
            }
        }
    }

    private func confidenceColor(_ confidence: Float) -> Color {
        if confidence > 0.8 { return .green }
        if confidence > 0.5 { return .orange }
        return .red
    }

    private func loadTestImage() {
        if let img = UIImage(named: "2") ?? UIImage(named: "2.png") {
            viewModel.selectedImage = img
            return
        }
        if let url = Bundle.main.url(forResource: "2", withExtension: "png"),
           let data = try? Data(contentsOf: url),
           let img = UIImage(data: data) {
            viewModel.selectedImage = img
            return
        }
        viewModel.errorMessage = "Couldn't find test image. Add it to Assets or Copy Bundle Resources."
    }
}

