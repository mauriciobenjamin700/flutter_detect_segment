import 'package:flutter/services.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../schemas/result.dart';
import './image_handler.dart';

/// Classe responsável pelo gerenciamento e execução de modelos TensorFlow Lite
/// para classificação de imagens.
///
/// Esta classe encapsula toda a lógica necessária para:
/// - Carregar modelos TFLite e arquivos de labels
/// - Executar inferência em imagens
/// - Processar e retornar resultados de classificação
class TFLiteHandler {
  /// Caminho para o arquivo do modelo TFLite
  final String modelPath;

  /// Caminho para o arquivo de labels/classes
  final String labelsPath;

  /// Instância do interpretador TensorFlow Lite
  Interpreter? _interpreter;

  /// Lista de labels/classes do modelo
  List<String>? _labels;

  /// Flag indicando se o modelo foi carregado com sucesso
  bool _isModelLoaded = false;

  /// Construtor da classe TFLiteHandler
  ///
  /// [modelPath] - Caminho para o arquivo .tflite do modelo
  /// [labelsPath] - Caminho para o arquivo .txt contendo os labels
  TFLiteHandler({required this.modelPath, required this.labelsPath});

  /// Carrega o modelo TensorFlow Lite e os arquivos de labels.
  ///
  /// Este método:
  /// - Verifica se o modelo já foi carregado para evitar recarregamento desnecessário
  /// - Carrega o modelo TFLite dos assets
  /// - Carrega e processa o arquivo de labels
  /// - Exibe informações de debug sobre o modelo
  ///
  /// Throws [Exception] se houver falha no carregamento
  Future<void> loadModel() async {
    try {
      if (_isModelLoaded) {
        print('⚠️ Modelo já carregado. Reutilizando...');
        return;
      }

      _interpreter = await Interpreter.fromAsset(modelPath);

      if (_interpreter == null) {
        print('❌ Erro ao inicializar o interpretador');
        return;
      }

      print('✅ Modelo carregado com sucesso');

      final labelsData = await rootBundle.loadString(labelsPath);
      _labels = labelsData
          .split('\n')
          .where((label) => label.isNotEmpty)
          .toList();

      _isModelLoaded = true;

      // Informações do modelo para debug
      final inputShape = _interpreter!.getInputTensor(0).shape;
      final outputShape = _interpreter!.getOutputTensor(0).shape;

      print('✅ Modelo carregado com sucesso!');
      print('📥 Input shape: $inputShape');
      print('📤 Output shape: $outputShape');
      print('🏷️ Labels carregados: ${_labels!.length}');
    } catch (e) {
      print('❌ Erro ao carregar modelo: $e');
    }
  }

  /// Classifica uma imagem e retorna as probabilidades de cada classe.
  ///
  /// [imagePath] - Caminho para a imagem a ser classificada (asset ou arquivo local)
  ///
  /// Returns: Lista de probabilidades para cada classe (já normalizadas pelo modelo)
  /// Returns: Lista vazia em caso de erro
  ///
  /// Throws: Exception em caso de falha na classificação
  Future<List<double>> classifyImage(String imagePath) async {
    if (!_isModelLoaded) {
      await loadModel();
    }
    if (_interpreter == null) {
      print('❌ Erro: Interprete não inicializado');
      return [];
    }

    try {
      final inputTensor = await ImageHandler.imageToTensor(
        imagePath,
        _interpreter!.getInputTensor(0).shape[2],
        _interpreter!.getInputTensor(0).shape[1],
      );

      final outputShape = _interpreter!.getOutputTensor(0).shape;
      final output = List.generate(
        outputShape[0],
        (index) => List<double>.filled(outputShape[1], 0.0),
      );

      _interpreter!.run(inputTensor, output);

      // ✅ Usa valores diretos (modelo já aplica softmax)
      final probabilities = output[0];

      // 🎨 Log formatado com 2 casas decimais
      final formattedProbs = probabilities
          .map((prob) => '${(prob * 100).toStringAsFixed(2)}%')
          .toList();

      print('🎯 Probabilidades: $formattedProbs');
      return probabilities;
    } catch (e) {
      print('❌ Erro ao classificar imagem: $e');
      return [];
    }
  }

  /// Classifica uma imagem e retorna o resultado completo com label e confiança.
  ///
  /// Este método combina a classificação com o processamento dos resultados,
  /// retornando a classe mais provável junto com sua confiança.
  ///
  /// [imagePath] - Caminho para a imagem a ser classificada
  ///
  /// Returns: [PredictionResult] contendo ID, label e confiança da predição
  /// Returns: Resultado com "Desconhecido" em caso de erro
  Future<PredictionResult> classifyImageWithLabels(String imagePath) async {
    final probabilities = await classifyImage(imagePath);

    // Encontra a classe com maior probabilidade
    double maxProb = 0.0;
    int maxIndex = 0;

    for (int i = 0; i < probabilities.length; i++) {
      if (probabilities[i] > maxProb) {
        maxProb = probabilities[i];
        maxIndex = i;
      }
    }

    final label = _labels != null && maxIndex < _labels!.length
        ? _labels![maxIndex]
        : 'Desconhecido';

    return PredictionResult(id: maxIndex, label: label, confidence: maxProb);
  }
  /// Retorna informações detalhadas sobre o modelo carregado.
  ///
  /// Inclui informações como:
  /// - Dimensões de entrada e saída do modelo
  /// - Número de labels/classes
  /// - Lista completa de labels
  ///
  /// Returns: Map com informações do modelo ou erro se não inicializado
  Map<String, dynamic> getModelInfo() {
    if (!_isModelLoaded) {
      return {'error': 'Modelo não inicializado'};
    }

    final inputShape = _interpreter!.getInputTensor(0).shape;
    final outputShape = _interpreter!.getOutputTensor(0).shape;

    return {
      'input_shape': inputShape,
      'output_shape': outputShape,
      'labels_count': _labels?.length ?? 0,
      'labels': _labels,
    };
  }

  /// Libera todos os recursos utilizados pelo interpretador TensorFlow Lite.
  ///
  /// Este método deve ser chamado quando o classificador não for mais utilizado
  /// para evitar vazamentos de memória. Após chamar este método, é necessário
  /// recarregar o modelo para utilizá-lo novamente.
  ///
  /// Limpa:
  /// - Instância do interpretador
  /// - Lista de labels
  /// - Flag de modelo carregado
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    _labels = null;
    _isModelLoaded = false;
    print('🧹 Recursos do TFLite liberados');
  }


  Future<List<SegmentationResult>> segmentImage(String imagePath, bool isFloat) async {
    if (!_isModelLoaded) {
      await loadModel();
    }
    if (_interpreter == null) {
      print('❌ Erro: Interprete não inicializado');
      return [];
    }

    try {
      final inputH = _interpreter!.getInputTensor(0).shape[1];
      final inputW = _interpreter!.getInputTensor(0).shape[2];

      final inputTensor = await ImageHandler.imageToTensor(
        imagePath,
        inputW,
        inputH,
      );

      final outTensor = _interpreter!.getOutputTensor(0);
      final outShape = outTensor.shape; // esperado: [1, H, W, C] ou [1, C, H, W] (às vezes [H, W, C])

      if (outShape.length < 3 || outShape.length > 4) {
        print('❌ Shape de saída não suportado: $outShape');
        return [];
      }

      // Aloca saída com o shape exato que o modelo retorna
      dynamic allocateOutput(List<int> shape, bool asFloat) {
        dynamic create(List<int> dims, int idx) {
          if (idx == dims.length - 1) {
            return asFloat
                ? List<double>.filled(dims[idx], 0.0)
                : List<int>.filled(dims[idx], 0);
          }
          return List.generate(dims[idx], (_) => create(dims, idx + 1));
        }
        return create(shape, 0);
      }

      final output = allocateOutput(outShape, isFloat);

      // Executa inferência
      _interpreter!.run(inputTensor, output);

      // Descobre layout e dimensões úteis
      late bool isNHWC;
      late int height;
      late int width;
      late int channels;

      if (outShape.length == 4) {
        // [N, H, W, C] ou [N, C, H, W]
        final labelsCount = _labels?.length;
        if (labelsCount != null && labelsCount > 1) {
          if (outShape[3] == labelsCount) {
            isNHWC = true;
          } else if (outShape[1] == labelsCount) {
            isNHWC = false;
          } else {
            isNHWC = outShape[1] >= 8 && outShape[2] >= 8; // heurística
          }
        } else {
          isNHWC = outShape[1] >= 8 && outShape[2] >= 8; // heurística
        }

        if (isNHWC) {
          height = outShape[1];
          width = outShape[2];
          channels = outShape[3];
        } else {
          channels = outShape[1];
          height = outShape[2];
          width = outShape[3];
        }
      } else {
        // [H, W, C]
        isNHWC = true;
        height = outShape[0];
        width = outShape[1];
        channels = outShape[2];
      }

      // Caso binário (1 canal) – gera máscara do "foreground"
      if (channels == 1) {
        final mask = List.generate(height, (_) => List<int>.filled(width, 0));
        int positive = 0;
        final total = height * width;

        if (outShape.length == 4) {
          if (isNHWC) {
            for (int y = 0; y < height; y++) {
              for (int x = 0; x < width; x++) {
                final v = isFloat
                    ? (output[0][y][x][0] as double)
                    : (output[0][y][x][0] as int).toDouble();
                final on = isFloat ? (v >= 0.5) : (v >= 128.0);
                mask[y][x] = on ? 1 : 0;
                if (on) positive++;
              }
            }
          } else {
            // NCHW
            for (int y = 0; y < height; y++) {
              for (int x = 0; x < width; x++) {
                final v = isFloat
                    ? (output[0][0][y][x] as double)
                    : (output[0][0][y][x] as int).toDouble();
                final on = isFloat ? (v >= 0.5) : (v >= 128.0);
                mask[y][x] = on ? 1 : 0;
                if (on) positive++;
              }
            }
          }
        } else {
          // [H, W, 1]
          for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              final v = isFloat
                  ? (output[y][x][0] as double)
                  : (output[y][x][0] as int).toDouble();
              final on = isFloat ? (v >= 0.5) : (v >= 128.0);
              mask[y][x] = on ? 1 : 0;
              if (on) positive++;
            }
          }
        }

        final confidence = total == 0 ? 0.0 : positive / total;
        final id = (_labels != null && _labels!.length >= 2) ? 1 : 0;
        final label = (_labels != null && _labels!.length > id)
            ? _labels![id]
            : 'foreground';

        print('🗺️ Segmentação binária concluída: $height x $width (p=${confidence.toStringAsFixed(3)})');
        return [
          SegmentationResult(
            id: id,
            label: label,
            confidence: confidence,
            mask: mask,
          )
        ];
      }

      // Multiclasse: prepara máscaras por classe
      final masks = List.generate(
        channels,
        (_) => List.generate(height, (_) => List<int>.filled(width, 0)),
      );
      final counts = List<int>.filled(channels, 0);
      final total = height * width;

      if (outShape.length == 4) {
        if (isNHWC) {
          // [1, H, W, C]
          for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              final vec = (output[0][y][x] as List);
              int bestIdx = 0;
              num bestVal = (vec[0] as num);
              for (int c = 1; c < channels; c++) {
                final v = (vec[c] as num);
                if (v > bestVal) {
                  bestVal = v;
                  bestIdx = c;
                }
              }
              masks[bestIdx][y][x] = 1;
              counts[bestIdx]++;
            }
          }
        } else {
          // [1, C, H, W]
          for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
              int bestIdx = 0;
              num bestVal = (output[0][0][y][x] as num);
              for (int c = 1; c < channels; c++) {
                final v = (output[0][c][y][x] as num);
                if (v > bestVal) {
                  bestVal = v;
                  bestIdx = c;
                }
              }
              masks[bestIdx][y][x] = 1;
              counts[bestIdx]++;
            }
          }
        }
      } else {
        // [H, W, C]
        for (int y = 0; y < height; y++) {
          for (int x = 0; x < width; x++) {
            final vec = (output[y][x] as List);
            int bestIdx = 0;
            num bestVal = (vec[0] as num);
            for (int c = 1; c < channels; c++) {
              final v = (vec[c] as num);
              if (v > bestVal) {
                bestVal = v;
                bestIdx = c;
              }
            }
            masks[bestIdx][y][x] = 1;
            counts[bestIdx]++;
          }
        }
      }

      // Constrói resultados apenas para classes presentes
      final results = <SegmentationResult>[];
      for (int c = 0; c < channels; c++) {
        if (counts[c] == 0) continue; // ignora classes ausentes
        final label = (_labels != null && c < _labels!.length)
            ? _labels![c]
            : 'Class $c';
        final confidence = total == 0 ? 0.0 : counts[c] / total; // proporção de área
        results.add(
          SegmentationResult(
            id: c,
            label: label,
            confidence: confidence,
            mask: masks[c],
          ),
        );
      }

      // Ordena por confiança (maior primeiro)
      results.sort((a, b) => b.confidence.compareTo(a.confidence));

      print('🗺️ Segmentação multiclasse concluída: $height x $width (classes=${results.length}/${channels})');
      return results;
    } catch (e) {
      print('❌ Erro ao segmentar imagem: $e');
      return [];
    }
  }


}