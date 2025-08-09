import 'dart:math' as math;

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


  double _sigmoid(num x) => 1.0 / (1.0 + math.exp(-x));
  int _clampInt(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);

  /// Upsample nearest 2D (rápido e suficiente para protótipos 160->640)
  List<List<double>> _upsampleNearest(List<List<double>> src, int outH, int outW) {
    final inH = src.length;
    final inW = inH > 0 ? src[0].length : 0;
    final out = List.generate(outH, (_) => List<double>.filled(outW, 0.0));
    for (int y = 0; y < outH; y++) {
      final sy = ((y * inH) / outH).floor();
      for (int x = 0; x < outW; x++) {
        final sx = ((x * inW) / outW).floor();
        out[y][x] = src[sy][sx];
      }
    }
    return out;
  }

  /// Gera máscara composta: sum_k coeff[k] * P_k e aplica sigmoid
  List<List<double>> _composeProtoMask(
    List<List<List<double>>> protos, // [K][H][W]
    List<double> coeffs,             // [K]
  ) {
    final k = protos.length;
    final h = k > 0 ? protos[0].length : 0;
    final w = (h > 0) ? protos[0][0].length : 0;
    final m = List.generate(h, (_) => List<double>.filled(w, 0.0));
    for (int c = 0; c < k; c++) {
      final pc = protos[c];
      final cv = coeffs[c];
      for (int y = 0; y < h; y++) {
        final row = pc[y];
        final dst = m[y];
        for (int x = 0; x < w; x++) {
          dst[x] += row[x] * cv;
        }
      }
    }
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        m[y][x] = _sigmoid(m[y][x]);
      }
    }
    return m;
  }

  /// Decodifica YOLO-seg a partir de:
  /// - dets: [1, 116, 8400] (4 box + 80 classes + 32 coef)
  /// - protos: [1, 32, 160, 160] ou [1, 160, 160, 32] ou [1, 32, Hm, Wm]
  Future<List<SegmentationResult>> _segmentImageYolo(
    dynamic dets,
    List<int> detShape,
    dynamic protos,
    List<int> protoShape,
    int inputH,
    int inputW,
  ) async {
    // Parse shapes
    final int feats = detShape[1];      // 116
    final int numAnchors = detShape[2]; // 8400

    // Índices esperados (4 box + 80 cls + 32 coef = 116)
    const int boxDims = 4;
    final int numClasses = (_labels?.length ?? 80);
    final int maskDims = feats - boxDims - numClasses; // deve dar 32

    // Extrai protótipos para formato [K][Hm][Wm]
    late int K, Hm, Wm;
    late List<List<List<double>>> P; // [K][Hm][Wm]
    if (protoShape.length == 4) {
      if (protoShape[1] == maskDims) {
        // [1, K, Hm, Wm] (NCHW)
        K = protoShape[1];
        Hm = protoShape[2];
        Wm = protoShape[3];
        P = List.generate(K, (c) {
          return List.generate(Hm, (y) {
            final row = List<double>.filled(Wm, 0.0);
            for (int x = 0; x < Wm; x++) {
              row[x] = (protos[0][c][y][x] as num).toDouble();
            }
            return row;
          });
        });
      } else if (protoShape[3] == maskDims) {
        // [1, Hm, Wm, K] (NHWC)
        Hm = protoShape[1];
        Wm = protoShape[2];
        K = protoShape[3];
        P = List.generate(K, (c) {
          return List.generate(Hm, (y) {
            final row = List<double>.filled(Wm, 0.0);
            for (int x = 0; x < Wm; x++) {
              row[x] = (protos[0][y][x][c] as num).toDouble();
            }
            return row;
          });
        });
      } else {
        print('❌ Shape de protótipos inesperado: $protoShape (maskDims=$maskDims)');
        return [];
      }
    } else {
      print('❌ Proto tensor deve ter rank 4. Recebido: $protoShape');
      return [];
    }

    // Thresholds simples
    const double confThresh = 0.25;
    const int maxDet = 20;

    // Coleta detecções
    final detsPicked = <Map<String, dynamic>>[];
    for (int i = 0; i < numAnchors; i++) {
      // Ordem típica (export YOLOv8/11-seg TFLite): [4 box][numClasses][maskDims], empilhados em dets[0][feat][i]
      final cx = (dets[0][0][i] as num).toDouble();
      final cy = (dets[0][1][i] as num).toDouble();
      final w  = (dets[0][2][i] as num).toDouble();
      final h  = (dets[0][3][i] as num).toDouble();

      // Classes
      int bestCls = 0;
      double bestScore = (dets[0][boxDims + 0][i] as num).toDouble();
      for (int c = 1; c < numClasses; c++) {
        final sc = (dets[0][boxDims + c][i] as num).toDouble();
        if (sc > bestScore) {
          bestScore = sc;
          bestCls = c;
        }
      }
      if (bestScore < confThresh) continue;

      // Coeficientes de máscara
      final coeff = <double>[];
      final coeffStart = boxDims + numClasses;
      for (int k = 0; k < maskDims; k++) {
        coeff.add((dets[0][coeffStart + k][i] as num).toDouble());
      }

      // Converte xywh -> xyxy
      final double x1 = cx - w / 2.0;
      final double y1 = cy - h / 2.0;
      final double x2 = cx + w / 2.0;
      final double y2 = cy + h / 2.0;

      detsPicked.add({
        'cls': bestCls,
        'score': bestScore,
        'xyxy': [x1, y1, x2, y2],
        'coeff': coeff,
      });
      if (detsPicked.length >= maxDet) break;
    }

    // Gera resultados (sem NMS para simplicidade)
    final results = <SegmentationResult>[];
    for (final d in detsPicked) {
      final List<double> coeff = (d['coeff'] as List<double>);
      // Composição no espaço dos protótipos
      final composed = _composeProtoMask(P, coeff);           // [Hm][Wm]
      final up = _upsampleNearest(composed, inputH, inputW);  // [H][W]

      // Recorte pela bbox e binarização
      final xyxy = (d['xyxy'] as List<double>);
      int x1 = _clampInt(xyxy[0].round(), 0, inputW - 1);
      int y1 = _clampInt(xyxy[1].round(), 0, inputH - 1);
      int x2 = _clampInt(xyxy[2].round(), 0, inputW - 1);
      int y2 = _clampInt(xyxy[3].round(), 0, inputH - 1);
      if (x2 <= x1 || y2 <= y1) continue;

      final mask = List.generate(inputH, (_) => List<int>.filled(inputW, 0));
      int onCount = 0;
      for (int y = y1; y <= y2; y++) {
        final row = up[y];
        final dst = mask[y];
        for (int x = x1; x <= x2; x++) {
          final v = row[x];
          if (v >= 0.5) {
            dst[x] = 1;
            onCount++;
          }
        }
      }

      final cls = d['cls'] as int;
      final label = (_labels != null && cls < _labels!.length) ? _labels![cls] : 'Class $cls';
      final score = (d['score'] as double);
      final area = (y2 - y1 + 1) * (x2 - x1 + 1);
      final confidence = area > 0 ? onCount / area : 0.0; // proporção ON dentro da bbox

      results.add(
        SegmentationResult(
          id: cls,
          label: label,
          confidence: score, // use score como confiança principal
          mask: mask,
        ),
      );
    }

    // Ordena por score
    results.sort((a, b) => b.confidence.compareTo(a.confidence));
    return results;
  }

  /// Segmentação unificada: trata YOLO-seg ([1,116,8400] + protótipos) e fallback denso [H,W,C].
  Future<List<SegmentationResult>> segmentImage(String imagePath, {double confThreshold = 0.25}) async {
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

      // Saída 0
      final out0 = _interpreter!.getOutputTensor(0);
      final out0Shape = out0.shape;

      // Tenta obter possível 2ª saída (prototypes)
      dynamic out1;
      List<int>? out1Shape;
      bool hasSecondOutput = false;
      try {
        final t1 = _interpreter!.getOutputTensor(1);
        out1Shape = t1.shape;
        hasSecondOutput = true;
      } catch (_) {
        hasSecondOutput = false;
      }

      // Caso YOLO-seg: [1, 116, 8400] + protótipos
      final looksLikeYoloSeg = (out0Shape.length == 3 && out0Shape[1] >= 100 && out0Shape[2] >= 3000);

      if (looksLikeYoloSeg) {
        if (!hasSecondOutput) {
          print('❌ Modelo parece YOLO-seg ([1,116,8400]) mas não há tensor de protótipos (saída 1).');
          return [];
        }

        // Alocação das saídas
        dynamic allocate(List<int> shape) {
          dynamic create(List<int> dims, int idx) {
            if (idx == dims.length - 1) {
              return List<double>.filled(dims[idx], 0.0);
            }
            return List.generate(dims[idx], (_) => create(dims, idx + 1));
          }
          return create(shape, 0);
        }

        final detsOut = allocate(out0Shape);
        out1 = allocate(out1Shape!);

        // Executa com múltiplas saídas
        _interpreter!.runForMultipleInputs([inputTensor], {0: detsOut, 1: out1});

        // Decodifica YOLO-seg
        return await _segmentImageYolo(
          detsOut,
          out0Shape,
          out1,
          out1Shape!,
          inputH,
          inputW,
        );
      }

      // Fallback: mapas densos (semântico) – seu código anterior pode permanecer aqui
      // ...existing code...

      print('❌ Saída não reconhecida para segmentação: $out0Shape');
      return [];
    } catch (e) {
      print('❌ Erro ao segmentar imagem: $e');
      return [];
    }
  }



}