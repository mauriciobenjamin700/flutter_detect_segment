class PredictionResult {
  final int id;
  final String label;
  final double confidence;

  PredictionResult({
    required this.id,
    required this.label,
    required this.confidence,
  });

  @override
  String toString() {
    return 'PredictionResult(id: $id, confidence: $confidence, label: $label)';
  }
}

/// Segmentation Result data
/// Attributes:
/// - id: Unique identifier for the result
/// - label: Class label for the segmentation
/// - confidence: Confidence score for the segmentation
/// - mask: Segmentation mask (binary mask indicating the segmented area)
/// Methods:
/// - toString: Returns a string representation of the segmentation result
class SegmentationResult {
  final int id;
  final String label;
  final double confidence;
  final List<List<int>> mask;

  SegmentationResult({
    required this.id,
    required this.label,
    required this.confidence,
    required this.mask,
  });

  @override
  String toString() {
    return 'SegmentationResult(id: $id, label: $label, confidence: $confidence, mask: $mask)';
  }
}