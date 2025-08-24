import 'package:flutter/material.dart';
import 'dart:io';
import '../core/constants.dart';
import '../services/image_segmentation.dart';
import '../widgets/image_gallery.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String? _selectedImage;
  List<List<double>>? _segmentationResult;
  final _imageSegmentationService = ImageSegmentationService();

  Future<void> _onImageSelected(String imagePath) async {
    // Atualiza imediatamente com a imagem selecionada da galeria
    setState(() {
      _selectedImage = imagePath;
    });

    // Após a segmentação, atualiza com o caminho da melhor máscara (se houver)
    final bestMaskPath = await _imageSegmentationService.segmentImage(
      imagePath,
    );
    if (bestMaskPath != null) {
      setState(() {
        _selectedImage = bestMaskPath;
      });
    }

    // final result = await _imageSegmentationService.segmentImage(ImagePath);
    // setState(() {
    //   _segmentationResult = result;
    // });
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
          // Preview da imagem selecionada (base para sobrepor segmentação futuramente)
          Padding(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
            child: Container(
              width: double.infinity,
              height: 240,
              decoration: BoxDecoration(
                color: Colors.grey.shade100,
                borderRadius: BorderRadius.circular(8),
                border: Border.all(color: Colors.grey.shade300),
              ),
              child: _selectedImage != null
                  ? ClipRRect(
                      borderRadius: BorderRadius.circular(8),
                      child: _selectedImage!.startsWith('assets/')
                          ? Image.asset(_selectedImage!, fit: BoxFit.contain)
                          : Image.file(
                              File(_selectedImage!),
                              fit: BoxFit.contain,
                            ),
                    )
                  : Center(
                      child: Text(
                        'Pré-visualização: selecione uma imagem acima',
                        style: TextStyle(color: Colors.grey.shade600),
                      ),
                    ),
            ),
          ),
        ],
      ),
    );
  }
}
