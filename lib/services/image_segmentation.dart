import '../core/tflite.dart';
import '../core/constants.dart';


class ImageSegmentationService {

  final TFLiteHandler _tflite = TFLiteHandler(
    modelPath: Constants.segmentModel,
    labelsPath: Constants.segmentLabels,
  );
  TFLiteHandler get tflite => _tflite;

  Future<void> segmentImage(String imagePath) async {

    await tflite.loadModel();

    await tflite.segmentImage(imagePath);
  }
}