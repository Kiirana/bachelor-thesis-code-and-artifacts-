//
//  BatchEvaluationView.swift
//  CurbDetectorApp
//
//  UI for running batch evaluation over a dataset of images.
//  Displays progress, per-stage latency stats, and export path.
//

import SwiftUI

struct BatchEvaluationView: View {
    @StateObject private var viewModel = BatchEvaluationViewModel()

    var body: some View {
        ScrollView {
            VStack(spacing: 20) {

                // MARK: - Header
                VStack(spacing: 8) {
                    Image(systemName: "chart.bar.doc.horizontal")
                        .font(.system(size: 40))
                        .foregroundColor(.indigo)
                    Text("Batch Latency Evaluation")
                        .font(.title2).fontWeight(.bold)
                    Text("Runs the full pipeline on a dataset and collects per-stage timing for the thesis latency table.")
                        .font(.caption)
                        .foregroundColor(.secondary)
                        .multilineTextAlignment(.center)
                        .padding(.horizontal)
                }

                // MARK: - Run button
                Button {
                    viewModel.runBatchEvaluation()
                } label: {
                    HStack {
                        if viewModel.isRunning {
                            ProgressView()
                                .progressViewStyle(CircularProgressViewStyle(tint: .white))
                        } else {
                            Image(systemName: "play.fill")
                        }
                        Text(viewModel.isRunning ? "Running…" : "Run Batch Evaluation")
                    }
                    .font(.headline)
                    .foregroundColor(.white)
                    .frame(maxWidth: .infinity)
                    .padding()
                    .background(viewModel.isRunning ? Color.gray : Color.indigo)
                    .cornerRadius(12)
                }
                .disabled(viewModel.isRunning)

                // MARK: - Progress
                if viewModel.isRunning {
                    VStack(spacing: 8) {
                        ProgressView(value: viewModel.progress) {
                            Text(viewModel.currentImage)
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }
                        Text("\(Int(viewModel.progress * 100))%")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color.indigo.opacity(0.05))
                    .cornerRadius(12)
                }

                // MARK: - Aggregated latency table
                if !viewModel.aggregatedStats.isEmpty {
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Latency Statistics")
                            .font(.headline)
                            .foregroundColor(.indigo)

                        // Header row
                        HStack {
                            Text("Stage").font(.caption2).fontWeight(.semibold).frame(width: 130, alignment: .leading)
                            Text("Mean").font(.caption2).fontWeight(.semibold).frame(width: 55, alignment: .trailing)
                            Text("Std").font(.caption2).fontWeight(.semibold).frame(width: 50, alignment: .trailing)
                            Text("Min").font(.caption2).fontWeight(.semibold).frame(width: 50, alignment: .trailing)
                            Text("Max").font(.caption2).fontWeight(.semibold).frame(width: 50, alignment: .trailing)
                        }
                        .padding(.horizontal, 4)

                        Divider()

                        ForEach(viewModel.aggregatedStats, id: \.stageName) { stat in
                            HStack {
                                Text(stat.stageName)
                                    .font(.caption2)
                                    .frame(width: 130, alignment: .leading)
                                    .lineLimit(1)
                                Text(String(format: "%.3f", stat.meanMs))
                                    .font(.system(.caption2, design: .monospaced))
                                    .frame(width: 55, alignment: .trailing)
                                Text(String(format: "%.3f", stat.stdMs))
                                    .font(.system(.caption2, design: .monospaced))
                                    .frame(width: 50, alignment: .trailing)
                                Text(String(format: "%.3f", stat.minMs))
                                    .font(.system(.caption2, design: .monospaced))
                                    .frame(width: 50, alignment: .trailing)
                                Text(String(format: "%.3f", stat.maxMs))
                                    .font(.system(.caption2, design: .monospaced))
                                    .frame(width: 50, alignment: .trailing)
                            }
                            .padding(.horizontal, 4)
                            .padding(.vertical, 2)
                            .background(stat.stageName == "total_e2e"
                                        ? Color.indigo.opacity(0.08)
                                        : Color.clear)
                            .cornerRadius(4)
                        }

                        Text("All values in milliseconds  •  n = \(viewModel.imageResults.count)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color.indigo.opacity(0.05))
                    .cornerRadius(12)
                }

                // MARK: - Per-image results (collapsible)
                if !viewModel.imageResults.isEmpty {

                    // Accuracy summary (only when ground-truth labels are present)
                    let labelled = viewModel.imageResults.filter { $0.groundTruth != nil && $0.textureClass != nil }
                    if !labelled.isEmpty {
                        let correct = labelled.filter { $0.isCorrect == true }.count
                        let acc = Double(correct) / Double(labelled.count)
                        VStack(spacing: 8) {
                            HStack {
                                Image(systemName: "checkmark.seal.fill")
                                    .foregroundColor(.green)
                                Text("Texture Accuracy")
                                    .font(.subheadline).fontWeight(.semibold)
                                Spacer()
                                Text(String(format: "%d/%d  (%.1f%%)", correct, labelled.count, acc * 100))
                                    .font(.subheadline.monospacedDigit())
                                    .foregroundColor(acc >= 0.9 ? .green : acc >= 0.75 ? .orange : .red)
                            }
                            let classes: [(String, Color)] = [
                                ("asphalt",     .gray),
                                ("cobblestone", .brown),
                                ("gravel",      .orange),
                                ("sand",        .yellow)
                            ]
                            ForEach(classes, id: \.0) { cls, col in
                                let forCls = labelled.filter { $0.groundTruth == cls }
                                if !forCls.isEmpty {
                                    let clsAcc = Double(forCls.filter { $0.isCorrect == true }.count) / Double(forCls.count)
                                    HStack(spacing: 6) {
                                        Text(cls)
                                            .font(.caption2)
                                            .frame(width: 76, alignment: .leading)
                                        GeometryReader { geo in
                                            ZStack(alignment: .leading) {
                                                RoundedRectangle(cornerRadius: 3)
                                                    .fill(Color.gray.opacity(0.15))
                                                RoundedRectangle(cornerRadius: 3)
                                                    .fill(col.opacity(0.7))
                                                    .frame(width: geo.size.width * clsAcc)
                                            }
                                        }
                                        .frame(height: 10)
                                        Text(String(format: "%.0f%%", clsAcc * 100))
                                            .font(.caption2.monospacedDigit())
                                            .frame(width: 34, alignment: .trailing)
                                    }
                                }
                            }
                        }
                        .padding()
                        .background(Color.green.opacity(0.05))
                        .cornerRadius(12)
                    }

                    DisclosureGroup {
                        LazyVStack(spacing: 4) {
                            HStack {
                                Text("File").font(.caption2).fontWeight(.semibold)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                Text("GT").font(.caption2).fontWeight(.semibold).frame(width: 62)
                                Text("Pred").font(.caption2).fontWeight(.semibold).frame(width: 68)
                                Text("Conf").font(.caption2).fontWeight(.semibold).frame(width: 38, alignment: .trailing)
                                Text("ms").font(.caption2).fontWeight(.semibold).frame(width: 46, alignment: .trailing)
                            }
                            .padding(.horizontal, 2)
                            Divider()
                            ForEach(viewModel.imageResults) { result in
                                HStack {
                                    Text(result.filename)
                                        .font(.caption2)
                                        .lineLimit(1)
                                        .frame(maxWidth: .infinity, alignment: .leading)
                                    Text(result.groundTruth?.capitalized ?? "—")
                                        .font(.caption2)
                                        .foregroundColor(.secondary)
                                        .frame(width: 62)
                                    HStack(spacing: 3) {
                                        Circle()
                                            .fill(result.isCorrect == true  ? Color.green
                                                : result.isCorrect == false ? Color.red
                                                : Color.gray.opacity(0.3))
                                            .frame(width: 6, height: 6)
                                        Text(result.textureClass?.capitalized ?? "—")
                                            .font(.caption2)
                                    }
                                    .frame(width: 68)
                                    Text(String(format: "%.0f%%", (result.textureConfidence ?? 0) * 100))
                                        .font(.system(.caption2, design: .monospaced))
                                        .foregroundColor(.secondary)
                                        .frame(width: 38, alignment: .trailing)
                                    Text(String(format: "%.0f", result.latency.totalMs))
                                        .font(.system(.caption2, design: .monospaced))
                                        .frame(width: 46, alignment: .trailing)
                                }
                                .padding(.vertical, 2)
                                .background(result.isCorrect == false ? Color.red.opacity(0.05) : Color.clear)
                            }
                        }
                    } label: {
                        Text("Per-Image Results (\(viewModel.imageResults.count))")
                            .font(.subheadline)
                            .fontWeight(.medium)
                    }
                    .padding()
                    .background(Color.gray.opacity(0.05))
                    .cornerRadius(12)
                }

                // MARK: - Export path
                if let path = viewModel.exportedFilePath {
                    VStack(alignment: .leading, spacing: 4) {
                        Label("Results exported", systemImage: "checkmark.circle.fill")
                            .font(.caption)
                            .foregroundColor(.green)
                        Text(path)
                            .font(.caption2)
                            .foregroundColor(.secondary)
                            .lineLimit(2)
                        Text("Also: latency_per_image.csv (same folder)")
                            .font(.caption2)
                            .foregroundColor(.secondary)
                    }
                    .padding()
                    .background(Color.green.opacity(0.05))
                    .cornerRadius(12)
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

                // MARK: - Instructions
                VStack(alignment: .leading, spacing: 8) {
                    Text("Setup Instructions")
                        .font(.caption).fontWeight(.semibold)
                    Text("1. Create a folder named \"EvalDataset\" with your test images (JPEG/PNG).")
                        .font(.caption2)
                    Text("2. In Xcode: Target → Build Phases → Copy Bundle Resources → add the folder as a folder reference (blue icon).")
                        .font(.caption2)
                    Text("3. Tap \"Run Batch Evaluation\".  Results (latency_stats.json + latency_per_image.csv) are saved to the Documents directory.")
                        .font(.caption2)
                    Text("Ground-truth (option A — JSON manifest): add \"100BatchrealTest.json\" (or \"labels.json\") to Copy Bundle Resources. Format: [{\"filename\": \"IMG_4531.HEIC\", \"groundTruth\": \"asphalt\"}]. Full macOS paths are also accepted — only the filename is matched.")
                        .font(.caption2)
                    Text("Ground-truth (option B — filename prefix): filenames must start with the class name followed by \"_\" (e.g. asphalt_01.jpg). The JSON manifest takes priority if present.")
                        .font(.caption2)
                    Text("4. Extract via Xcode: Window → Devices → Download Container, or use the Files app.")
                        .font(.caption2)
                }
                .foregroundColor(.secondary)
                .padding()
                .background(Color.gray.opacity(0.05))
                .cornerRadius(12)
            }
            .padding()
        }
        .navigationTitle("Batch Evaluation")
        .navigationBarTitleDisplayMode(.inline)
    }
}

