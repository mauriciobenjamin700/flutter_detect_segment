import '../core/tflite.dart';
import '../core/constants.dart';

class ImageSegmentationService {
  final TFLiteHandler _tflite = TFLiteHandler(
    modelPath: Constants.segmentModel,
    labelsPath: Constants.segmentLabels,
  );
  TFLiteHandler get tflite => _tflite;

  Future<String?> segmentImage(String imagePath) async {
    await tflite.loadModel();
    final bestMaskPath = await tflite.segmentImage(imagePath);
    if (bestMaskPath != null) {
      // imprime para debug
      // ignore: avoid_print
      print('Melhor máscara salva em: $bestMaskPath');
    }
    return bestMaskPath;
  }
}
