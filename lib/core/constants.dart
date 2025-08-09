Map<String, String> mockImages = {
  'person1': 'assets/person1.png',
  'person2': 'assets/person2.png',
  'person4': 'assets/person4.png',
  'person5': 'assets/person5.png',
};

class Constants {
  static const String segmentModel = 'assets/models/yolo11n-seg_float16.tflite';
  static const String segmentLabels = 'assets/models/labels.txt';
  static const int segmentImageWidth = 640;  // ou 224 se seu modelo usar
  static const int segmentImageHeight = 640; // ou 224 se seu modelo usar
}