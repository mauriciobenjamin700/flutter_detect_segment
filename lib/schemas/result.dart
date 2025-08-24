class ClassificationResult {
  final int id;
  final String label;
  final double confidence;

  ClassificationResult({
    required this.id,
    required this.label,
    required this.confidence,
  });

  @override
  String toString() {
    return 'ClassificationResult(id: $id, confidence: $confidence, label: $label)';
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
  final List<List<int>>? boundingBoxes; // [[x1,y1,x2,y2], [x1,y1,x2,y2],...]

  SegmentationResult({
    required this.id,
    required this.label,
    required this.confidence,
    required this.mask,
    required this.boundingBoxes
  });

  @override
  String toString() {
    return 'SegmentationResult(id: $id, label: $label, confidence: $confidence, mask: $mask , boundingBoxes: $boundingBoxes)';
  }
}
