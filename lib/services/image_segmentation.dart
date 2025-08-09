import '../core/tflite.dart';
import '../core/constants.dart';
import '../schemas/result.dart';

class ImageSegmentationService {

  final TFLiteHandler _tflite = TFLiteHandler(
    modelPath: Constants.segmentModel,
    labelsPath: Constants.segmentLabels,
  );
  TFLiteHandler get tflite => _tflite;

  Future<List<SegmentationResult>> segmentImage(String imagePath) async {

    await tflite.loadModel();

    final output = await tflite.segmentImage(imagePath, true);

    for (var result in output) {
      print('🔍 Resultado da segmentação: ${result.toString()}');
    }

    return output;
  }

}