import 'package:flutter/material.dart';
import '../core/constants.dart';
import '../schemas/result.dart';
import '../services/image_segmentation.dart';
import '../widgets/image_gallery.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? _selectedImage;
  List<SegmentationResult>? _segmentationResult;
  final _imageSegmentationService = ImageSegmentationService();

  Future<void> _onImageSelected(String ImagePath) async {
    setState(() {
      _selectedImage = ImagePath;
    });
    final result = await _imageSegmentationService.segmentImage(ImagePath);
    setState(() {
      _segmentationResult = result;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Home Page')),
      body: Column(
        children: [
          Container(
            width: double.infinity,
            padding: EdgeInsets.all(16.0),
            margin: EdgeInsets.all(16.0),
            decoration: BoxDecoration(
              color: Colors.blue.shade50,
              borderRadius: BorderRadius.circular(8),
              border: Border.all(color: Colors.blue.shade200),
            ),
            child: Text(
              _selectedImage != null
                  ? 'Selected Image: $_selectedImage'
                  : 'No image selected',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
                color: _selectedImage != null
                    ? Colors.blue.shade800
                    : Colors.grey.shade600,
              ),
              textAlign: TextAlign.center,
            ),
          ),
          Expanded(
            child: ImageGallery(
              imagePaths: [
                mockImages['person1']!,
                mockImages['person2']!,
                mockImages['person4']!,
                mockImages['person5']!,
              ],
              onImageTap: _onImageSelected,
            ),
          ),
          Expanded(
            child: Text(
              _segmentationResult != null
                  ? 'Segmentation Result: \n ${_segmentationResult?.map((e) => e.label).join(", ")}'
                  : 'No image selected',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w500,
                color: _segmentationResult != null
                    ? Colors.green.shade800
                    : Colors.grey.shade600,
              ),
              textAlign: TextAlign.center,
            ),
          ),
        ],
      ),
    );
  }
}