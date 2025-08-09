import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;

class ImageHandler {
  static Future<Uint8List> loadImage(String path) async {
    if (path.startsWith('assets/')) {
      final ByteData data = await rootBundle.load(path);
      return data.buffer.asUint8List();
    } else {
      // Para arquivos locais
      final file = File(path);
      return await file.readAsBytes();
    }
  }


  static Future<List<List<List<List<double>>>>> imageToTensor(
    String imagePath,
    int tensorWidth,
    int tensorHeight,
  ) async {
    try {
      final imageBytes = await loadImage(imagePath);
      final originalImage = img.decodeImage(imageBytes);
      
      if (originalImage == null) {
        throw Exception('Erro ao decodificar imagem: $imagePath');
      }

      final resizedImage = img.copyResize(
        originalImage,
        width: tensorWidth,
        height: tensorHeight,
      );

      // Inicializa tensor
      final tensor = List.generate(
        1,
        (_) => List.generate(
          tensorHeight,
          (_) => List.generate(
            tensorWidth,
            (_) => List.generate(3, (_) => 0.0),
          ),
        ),
      );

      // Preenche tensor usando iterador (forma correta)
      int pixelIndex = 0;
      for (final pixel in resizedImage) {
        final y = pixelIndex ~/ tensorWidth;
        final x = pixelIndex % tensorWidth;
        
        // Usa maxChannelValue para normalização automática
        tensor[0][y][x][0] = pixel.r / pixel.maxChannelValue; // Red
        tensor[0][y][x][1] = pixel.g / pixel.maxChannelValue; // Green
        tensor[0][y][x][2] = pixel.b / pixel.maxChannelValue; // Blue
        
        pixelIndex++;
      }

      print('📊 Tensor otimizado: [${tensor.length}, ${tensor[0].length}, ${tensor[0][0].length}, ${tensor[0][0][0].length}]');
      return tensor;
      
    } catch (e) {
      print('❌ Erro ao converter imagem para tensor: $e');
      rethrow;
    }
  }

  /// Versão que retorna Float32List (mais performática)
  static Future<Float32List> imageToFloat32List(
    String imagePath,
    int tensorWidth,
    int tensorHeight,
  ) async {
    try {
      final imageBytes = await loadImage(imagePath);
      final originalImage = img.decodeImage(imageBytes);
      
      if (originalImage == null) {
        throw Exception('Erro ao decodificar imagem: $imagePath');
      }

      final resizedImage = img.copyResize(
        originalImage,
        width: tensorWidth,
        height: tensorHeight,
      );

      final tensorSize = 1 * tensorHeight * tensorWidth * 3;
      final tensor = Float32List(tensorSize);
      
      int index = 0;
      for (final pixel in resizedImage) {
        tensor[index++] = pixel.r / pixel.maxChannelValue; // Red
        tensor[index++] = pixel.g / pixel.maxChannelValue; // Green
        tensor[index++] = pixel.b / pixel.maxChannelValue; // Blue
      }

      print('📊 Float32List criado: ${tensor.length} elementos');
      return tensor;
      
    } catch (e) {
      print('❌ Erro ao converter imagem para Float32List: $e');
      rethrow;
    }
  }

  /// Para debug - mostra informações da imagem
  static Future<void> debugImageInfo(String imagePath) async {
    final imageBytes = await loadImage(imagePath);
    final image = img.decodeImage(imageBytes);
    
    if (image != null) {
      print('🖼️ Informações da imagem:');
      print('   - Dimensões: ${image.width}x${image.height}');
      print('   - Formato: ${image.format}');
      print('   - Canais: ${image.numChannels}');
      print('   - Max channel value: ${image.maxChannelValue}');
    }
  }
}