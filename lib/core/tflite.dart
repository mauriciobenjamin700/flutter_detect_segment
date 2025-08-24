import 'dart:math' as math;
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import '../schemas/result.dart';
import './image_handler.dart';

/// Classe responsável pelo gerenciamento e execução de modelos TensorFlow Lite
class TFLiteHandler {
  final String modelPath;
  final String labelsPath;

  Interpreter? _interpreter;
  List<String>? _labels;
  bool _isModelLoaded = false;

  TFLiteHandler({required this.modelPath, required this.labelsPath});

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

      final labelsData = await rootBundle.loadString(labelsPath);
      _labels = labelsData
          .split('\n')
          .where((label) => label.trim().isNotEmpty)
          .toList();

      _isModelLoaded = true;

      final inputShape = _interpreter!.getInputTensor(0).shape;
      final output0Shape = _interpreter!.getOutputTensor(0).shape;
      String out1Info = 'n/a';
      try {
        final o1 = _interpreter!.getOutputTensor(1);
        out1Info = o1.shape.toString();
      } catch (_) {
        out1Info = 'não disponível';
      }

      print('✅ Modelo carregado com sucesso!');
      print('📥 Input shape: $inputShape');
      print('📤 Output[0] shape: $output0Shape');
      print('📤 Output[1] shape: $out1Info');
      print('🏷️ Labels carregados: ${_labels?.length ?? 0}');
    } catch (e) {
      print('❌ Erro ao carregar modelo: $e');
      rethrow;
    }
  }

  /// A função allocateOutput é responsável por criar uma estrutura de dados (listas aninhadas) que corresponde ao formato (ou shape) especificado. Essa estrutura é usada para armazenar os tensores de saída do modelo TensorFlow Lite.
  /// TensorFlow Lite exige que as saídas sejam pré-alocadas com o mesmo formato do tensor de saída do modelo.
  /// Essa função garante que a estrutura de saída tenha o formato correto, independentemente do número de dimensões ou do tamanho de cada dimensão.
  dynamic allocateOutput(List<int> shape) {
    dynamic create(int dim) {
      if (dim == shape.length - 1) {
        return List<double>.filled(shape[dim], 0.0);
      }
      return List.generate(shape[dim], (_) => create(dim + 1));
    }

    return create(0);
  }

  Future<List<double>> classifyImage(String imagePath) async {
    if (!_isModelLoaded) await loadModel();
    if (_interpreter == null) return [];

    try {
      final inMeta = _interpreter!.getInputTensor(0);
      final inputH = inMeta.shape.length > 1 ? inMeta.shape[1] : 0;
      final inputW = inMeta.shape.length > 2 ? inMeta.shape[2] : 0;

      final inputTensor = await ImageHandler.imageToTensor(
        imagePath,
        inputW,
        inputH,
      );

      final outShape = _interpreter!.getOutputTensor(0).shape;
      final output = allocateOutput(outShape);

      _interpreter!.run(inputTensor, output);

      // assume output shape [1, N]
      final probs = (output is List && output.isNotEmpty)
          ? List<double>.from(output[0].map((v) => (v as num).toDouble()))
          : <double>[];
      return probs;
    } catch (e) {
      print('❌ Erro ao classificar imagem: $e');
      return [];
    }
  }

  /// Segmenta uma imagem usando o modelo carregado.
  /// **1**: Representa o batch size (processamento de uma única imagem por vez).
  /// **feats**: Número de atributos por detecção. Geralmente inclui:
  /// 4 valores para bounding boxes: (cx, cy, w, h).
  /// N valores para classes: Probabilidades para cada classe (ex.: 80 classes no COCO).
  /// K valores para coeficientes de máscara: Usados para compor as máscaras de segmentação.
  /// Exemplo: feats = 4 (bounding boxes) + 80 (classes) + 32 (coeficientes de máscara) = 116.
  /// **anchors**: Número total de detecções (ou "anchors") que o modelo processa. Exemplo: 8400 anchors.
  /// **K**: Número de protótipos (ou dimensões latentes) usados para compor as máscaras de segmentação. Exemplo: 32.
  /// Hm e Wm: Altura e largura do mapa de protótipos (geralmente menor que a resolução da imagem de entrada, ex.: 160x160).
  Future<void> segmentImage(
    String imagePath, {
    double confThreshold = 0.25,
  }) async {
    if (!_isModelLoaded) await loadModel();

    try {
      debugPrint('🔍 Iniciando segmentação para: $imagePath');
      if (_interpreter == null) {
        debugPrint('❌ Interpretador TFLite não inicializado');
        return;
      }

      // 1) Metadados de entrada
      final inTensor = _interpreter!.getInputTensor(0);
      final inShape = inTensor.shape; // esperado [1, H, W, C]
      final inputH = inShape.length > 2 ? inShape[1] : 0;
      final inputW = inShape.length > 2 ? inShape[2] : 0;
      final inputC = inShape.length > 3 ? inShape[3] : 0;
      debugPrint('📥 Input[0] shape=$inShape type=${inTensor.type}');

      // 2) Pré-processamento (usa seu helper existente)
      final inputTensor = await ImageHandler.imageToTensor(
        imagePath,
        inputW,
        inputH,
      );
      debugPrint(
        '🧪 Input tensor pronto (C=$inputC) -> tipo=${inputTensor.runtimeType}',
      );

      // 3) Descobrir todas as saídas disponíveis
      final outputShapes = <int, List<int>>{};
      final outputTypes = <int, Object>{};
      int outIndex = 0;
      while (true) {
        try {
          final t = _interpreter!.getOutputTensor(outIndex);
          outputShapes[outIndex] = t.shape;
          outputTypes[outIndex] = t.type;
          outIndex++;
        } catch (_) {
          break;
        }
      }
      if (outputShapes.isEmpty) {
        debugPrint('❌ Nenhuma saída encontrada no modelo');
        return;
      }
      for (final i in outputShapes.keys) {
        debugPrint(
          '📤 Output[$i] shape=${outputShapes[i]} type=${outputTypes[i]}',
        );
      }

      // 4) Alocar objetos de saída conforme os shapes
      final outputs = <int, dynamic>{};
      outputShapes.forEach((i, shape) {
        outputs[i] = allocateOutput(shape);
      });

      // 5) Executar inferência (múltiplas saídas se houver)
      if (outputs.length == 1) {
        final onlyOut = outputs.values.first;
        _interpreter!.run(inputTensor, onlyOut);
        debugPrint('✅ run() concluído');
      } else {
        _interpreter!.runForMultipleInputs([
          inputTensor,
        ], outputs.map((k, v) => MapEntry(k, v)));
        debugPrint(
          '✅ runForMultipleInputs() concluído (${outputs.length} saídas)',
        );
      }

      // 6) Debug print de todas as saídas (shape + valores)
      const maxElementsToPrint = 100; // limite para não travar o log
      for (final i in outputs.keys.toList()..sort()) {
        final data = outputs[i];
        final shape = outputShapes[i]!;
        debugPrint('=== 🔎 Dump Output[$i] (shape=$shape) ===');
        final flat = _flattenToNumList(data);
        _printFlatList(flat, maxElements: maxElementsToPrint);
        if (flat.length > maxElementsToPrint) {
          debugPrint(
            '... (${flat.length - maxElementsToPrint} valores não impressos)',
          );
        }
      }

      // 7) Fim
      debugPrint('🏁 Segmentação concluída (debug de saídas impresso).');
    } catch (e) {
      print('❌ Erro ao segmentar imagem: $e');
      return;
    }
  }

  // Auxiliares de debug
  List<num> _flattenToNumList(dynamic data) {
    final out = <num>[];
    void walk(dynamic x) {
      if (x is List) {
        for (final v in x) {
          walk(v);
        }
      } else if (x is num) {
        out.add(x);
      } else {
        // ignora tipos não numéricos
      }
    }

    walk(data);
    return out;
  }

  void _printFlatList(List<num> list, {int maxElements = 8000}) {
    final n = math.min(list.length, maxElements);
    final buf = StringBuffer();
    for (int i = 0; i < n; i++) {
      buf.write(list[i]);
      if (i + 1 < n) buf.write(', ');
      if ((i + 1) % 64 == 0) buf.writeln();
    }
    debugPrint(buf.toString());
  }
}
