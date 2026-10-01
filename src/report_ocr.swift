import Foundation
import Vision
let url = URL(fileURLWithPath: CommandLine.arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
let handler = VNImageRequestHandler(url: url, options: [:])
do {
    try handler.perform([request])
    for observation in request.results ?? [] {
        if let text = observation.topCandidates(1).first { print(text.string) }
    }
} catch {
    FileHandle.standardError.write(Data(String(describing: error).utf8))
    exit(1)
}
