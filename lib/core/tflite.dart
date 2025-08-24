import 'dart:math' as math;
import 'dart:io';
import 'dart:typed_data';
import 'package:image/image.dart' as img;
import 'package:flutter/services.dart';
import 'package:flutter/widgets.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import './image_handler.dart';

/// Classe responsável pelo gerenciamento e execução de modelos TensorFlow Lite
class TFLiteHandler {
  final String modelPath;
  final String labelsPath;

  Interpreter? _interpreter;
  IsolateInterpreter? _isolateInterpreter;
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

      _isolateInterpreter = await IsolateInterpreter.create(
        address: _interpreter!.address,
      );

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

      await _isolateInterpreter!.run(inputTensor, output);

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
  Future<String?> segmentImage(
    String imagePath, {
    double confThreshold = 0.2,
  }) async {
    if (!_isModelLoaded) await loadModel();

    try {
      debugPrint('🔍 Iniciando segmentação para: $imagePath');
      if (_interpreter == null) {
        debugPrint('❌ Interpretador TFLite não inicializado');
        return null;
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
        return null;
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
        await _isolateInterpreter!.run(inputTensor, onlyOut);
        debugPrint('✅ run() concluído');
      } else {
        await _isolateInterpreter!.runForMultipleInputs([
          inputTensor,
        ], outputs.map((k, v) => MapEntry(k, v)));
        debugPrint(
          '✅ runForMultipleInputs() concluído (${outputs.length} saídas)',
        );
      }

      // 6) Debug print de todas as saídas (shape + valores)
      const maxElementsToPrint = 1; // limite para não travar o log
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

      // 7) Tenta extrair máscaras (YOLO-seg: out0 [1,116,anchors], out1 [1,Hm,Wm,32] ou [1,32,Hm,Wm])
      String? bestMaskPath;
      double bestMaskScore = -1.0;
      if (outputShapes.length >= 2) {
        final detShape = outputShapes[0]!;
        final protoShape = outputShapes[1]!;
        final isYoloSeg =
            detShape.length == 3 && detShape[1] >= 100 && detShape[2] >= 1000;
        if (isYoloSeg) {
          final dets = outputs[0];
          final protos = outputs[1];

          final feats = detShape[1];
          final anchors = detShape[2];
          final numClasses = _labels?.length ?? 80;
          const boxDims = 4;
          final maskDims = feats - boxDims - numClasses; // ex.: 32

          // Extrai protótipos no formato [K][Hm][Wm]
          late int K, Hm, Wm;
          late List<List<List<double>>> P;
          if (protoShape.length == 4 && protoShape[3] == maskDims) {
            // NHWC [1,Hm,Wm,K]
            Hm = protoShape[1];
            Wm = protoShape[2];
            K = protoShape[3];
            P = List.generate(K, (c) {
              return List.generate(Hm, (y) {
                return List.generate(
                  Wm,
                  (x) => (protos[0][y][x][c] as num).toDouble(),
                );
              });
            });
          } else if (protoShape.length == 4 && protoShape[1] == maskDims) {
            // NCHW [1,K,Hm,Wm]
            K = protoShape[1];
            Hm = protoShape[2];
            Wm = protoShape[3];
            P = List.generate(K, (c) {
              return List.generate(Hm, (y) {
                return List.generate(
                  Wm,
                  (x) => (protos[0][c][y][x] as num).toDouble(),
                );
              });
            });
          } else {
            debugPrint(
              '⚠️ Formato de protótipos não suportado: $protoShape (maskDims=$maskDims)',
            );
            debugPrint('🏁 Segmentação concluída (sem extração de máscara).');
            return null;
          }

          // Itera detecções, compõe e salva algumas máscaras
          const maxToSave = 5;
          int saved = 0;
          for (int i = 0; i < anchors; i++) {
            // Caixa: cx, cy, w, h
            double cx = (dets[0][0][i] as num).toDouble();
            double cy = (dets[0][1][i] as num).toDouble();
            double w = (dets[0][2][i] as num).toDouble();
            double h = (dets[0][3][i] as num).toDouble();

            // Alguns exports retornam caixas normalizadas [0,1]; autoescala se parecer o caso
            final maxBoxVal = [
              cx.abs(),
              cy.abs(),
              w.abs(),
              h.abs(),
            ].reduce(math.max);
            final boxesLikelyNormalized = maxBoxVal <= 1.5; // margem p/ ruído
            if (boxesLikelyNormalized) {
              cx *= inputW;
              w *= inputW;
              cy *= inputH;
              h *= inputH;
            }

            // Classe + score (pega a melhor)
            int bestCls = 0;
            // Aplica sigmoid nos logits de classe (YOLO v8-seg exporta logits)
            double bestScore = _sigmoid(
              (dets[0][boxDims + 0][i] as num).toDouble(),
            );
            for (int c = 1; c < numClasses; c++) {
              final raw = (dets[0][boxDims + c][i] as num).toDouble();
              final sc = _sigmoid(raw);
              if (sc > bestScore) {
                bestScore = sc;
                bestCls = c;
              }
            }
            if (bestScore < confThreshold) continue;

            // Coeficientes da máscara
            final coeff = <double>[];
            final coeffStart = boxDims + numClasses;
            for (int k = 0; k < maskDims; k++) {
              coeff.add((dets[0][coeffStart + k][i] as num).toDouble());
            }

            // BBox em xyxy
            final x1 = (cx - w / 2.0).round().clamp(0, inputW - 1);
            final y1 = (cy - h / 2.0).round().clamp(0, inputH - 1);
            final x2 = (cx + w / 2.0).round().clamp(0, inputW - 1);
            final y2 = (cy + h / 2.0).round().clamp(0, inputH - 1);
            if (x2 <= x1 || y2 <= y1) continue;

            // Composição -> upsample -> binarização
            final composed = _composeProtoMask(P, coeff); // [Hm][Wm]
            final up = _upsampleBilinear(composed, inputH, inputW); // [H][W]
            final mask = List.generate(
              inputH,
              (_) => List<int>.filled(inputW, 0),
            );
            int on = 0;
            const maskBinThreshold = 0.30; // menor para evitar tudo preto
            for (int y = y1; y <= y2; y++) {
              final srcRow = up[y];
              final dstRow = mask[y];
              for (int x = x1; x <= x2; x++) {
                if (srcRow[x] >= maskBinThreshold) {
                  dstRow[x] = 255; // branco
                  on++;
                }
              }
            }
            if (on == 0) continue;

            final label = (_labels != null && bestCls < _labels!.length)
                ? _labels![bestCls]
                : 'class_$bestCls';
            // Estatísticas para diagnosticar máscaras pretas
            double localMin = double.infinity;
            double localMax = -double.infinity;
            for (int y = y1; y <= y2; y++) {
              final row = up[y];
              for (int x = x1; x <= x2; x++) {
                final v = row[x];
                if (v < localMin) localMin = v;
                if (v > localMax) localMax = v;
              }
            }
            debugPrint(
              '🧩 det#$i label=$label conf=${bestScore.toStringAsFixed(3)} box=${x1},${y1},${x2},${y2} normBoxes=$boxesLikelyNormalized onPixels=$on local[min=${localMin.toStringAsFixed(3)}, max=${localMax.toStringAsFixed(3)}]',
            );
            final path = await _saveCutoutAsPng(imagePath, mask, suffix: label);
            debugPrint(
              '💾 Imagem segmentada salva em: $path (label=$label, conf=${bestScore.toStringAsFixed(3)})',
            );
            saved++;
            if (bestScore > bestMaskScore) {
              bestMaskScore = bestScore;
              bestMaskPath = path;
            }
            if (saved >= maxToSave) break;
          }
          if (saved == 0) {
            debugPrint(
              'ℹ️ Nenhuma máscara acima do limiar conf=${confThreshold.toStringAsFixed(2)}',
            );
          }
        }
      }

      // 8) Fim
      debugPrint(
        '🏁 Segmentação concluída (debug + tentativa de salvar máscaras).',
      );
      if (bestMaskPath != null) {
        debugPrint(
          '⭐ Melhor máscara: $bestMaskPath (score=${bestMaskScore.toStringAsFixed(3)})',
        );
      }
      return bestMaskPath;
    } catch (e) {
      print('❌ Erro ao segmentar imagem: $e');
      return null;
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

  // ===== Helpers para compor/upsample e salvar PNG =====
  double _sigmoid(num x) => 1.0 / (1.0 + math.exp(-x));

  List<List<double>> _composeProtoMask(
    List<List<List<double>>> protos,
    List<double> coeffs,
  ) {
    final K = protos.length;
    final H = K > 0 ? protos[0].length : 0;
    final W = (H > 0) ? protos[0][0].length : 0;
    final out = List.generate(H, (_) => List<double>.filled(W, 0.0));
    for (int k = 0; k < K; k++) {
      final pk = protos[k];
      final ck = coeffs[k];
      for (int y = 0; y < H; y++) {
        final src = pk[y];
        final dst = out[y];
        for (int x = 0; x < W; x++) {
          dst[x] += src[x] * ck;
        }
      }
    }
    for (int y = 0; y < out.length; y++) {
      for (int x = 0; x < out[y].length; x++) {
        out[y][x] = _sigmoid(out[y][x]);
      }
    }
    return out;
  }

  List<List<double>> _upsampleBilinear(
    List<List<double>> src,
    int outH,
    int outW,
  ) {
    final inH = src.length;
    final inW = inH > 0 ? src[0].length : 0;
    if (inH == 0 || inW == 0) {
      return List.generate(outH, (_) => List<double>.filled(outW, 0.0));
    }
    final out = List.generate(outH, (_) => List<double>.filled(outW, 0.0));
    final scaleY = (inH - 1) / (outH - 1);
    final scaleX = (inW - 1) / (outW - 1);
    for (int y = 0; y < outH; y++) {
      final fy = y * scaleY;
      final y0 = fy.floor();
      final y1 = math.min(y0 + 1, inH - 1);
      final wy = fy - y0;
      for (int x = 0; x < outW; x++) {
        final fx = x * scaleX;
        final x0 = fx.floor();
        final x1 = math.min(x0 + 1, inW - 1);
        final wx = fx - x0;

        final v00 = src[y0][x0];
        final v01 = src[y0][x1];
        final v10 = src[y1][x0];
        final v11 = src[y1][x1];

        final v0 = v00 * (1 - wx) + v01 * wx;
        final v1 = v10 * (1 - wx) + v11 * wx;
        out[y][x] = v0 * (1 - wy) + v1 * wy;
      }
    }
    return out;
  }

  Future<String> _saveCutoutAsPng(
    String originalPath,
    List<List<int>> mask, {
    String suffix = 'seg',
  }) async {
    // Carrega imagem original (asset ou arquivo)
    Uint8List bytes;
    if (originalPath.startsWith('assets/')) {
      final data = await rootBundle.load(originalPath);
      bytes = data.buffer.asUint8List();
    } else {
      bytes = await File(originalPath).readAsBytes();
    }
    final orig = img.decodeImage(bytes);
    if (orig == null) {
      throw Exception('decodeImage falhou para $originalPath');
    }

    // Ajusta tamanho para bater com a máscara (inferida no input do modelo)
    final h = mask.length;
    final w = h > 0 ? mask[0].length : 0;
    final base = (orig.width != w || orig.height != h)
        ? img.copyResize(orig, width: w, height: h)
        : orig;

    // Constrói imagem recortada com fundo transparente
    final cut = img.Image(width: w, height: h);
    for (int y = 0; y < h; y++) {
      for (int x = 0; x < w; x++) {
        if (mask[y][x] != 0) {
          final px = base.getPixel(x, y);
          cut.setPixelRgba(x, y, px.r, px.g, px.b, 255);
        } else {
          cut.setPixelRgba(x, y, 0, 0, 0, 0);
        }
      }
    }

    final outBytes = img.encodePng(cut);
    final dir = Directory.systemTemp;
    final file = File(
      '${dir.path}/seg_cutout_${suffix}_${DateTime.now().millisecondsSinceEpoch}.png',
    );
    await file.writeAsBytes(outBytes, flush: true);
    return file.path;
  }
}
