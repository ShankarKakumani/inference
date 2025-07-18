// This file is automatically generated, so please do not edit it.
// @generated by `flutter_rust_bridge`@ 2.11.1.

// ignore_for_file: invalid_use_of_internal_member, unused_import, unnecessary_import

import '../frb_generated.dart';
import 'package:flutter_rust_bridge/flutter_rust_bridge_for_generated.dart';
import 'package:freezed_annotation/freezed_annotation.dart' hide protected;
part 'error.freezed.dart';

@freezed
sealed class InferenceError with _$InferenceError implements FrbException {
  const InferenceError._();

  const factory InferenceError.modelLoad(
    String field0,
  ) = InferenceError_ModelLoad;
  const factory InferenceError.prediction(
    String field0,
  ) = InferenceError_Prediction;
  const factory InferenceError.unsupportedFormat(
    String field0,
  ) = InferenceError_UnsupportedFormat;
  const factory InferenceError.invalidShape({
    required Uint64List expected,
    required Uint64List actual,
  }) = InferenceError_InvalidShape;
  const factory InferenceError.invalidTensorData(
    String field0,
  ) = InferenceError_InvalidTensorData;
  const factory InferenceError.engine(
    String field0,
  ) = InferenceError_Engine;
  const factory InferenceError.io(
    String field0,
  ) = InferenceError_Io;
  const factory InferenceError.serialization(
    String field0,
  ) = InferenceError_Serialization;
  const factory InferenceError.configuration(
    String field0,
  ) = InferenceError_Configuration;
  const factory InferenceError.resourceNotFound(
    String field0,
  ) = InferenceError_ResourceNotFound;
  const factory InferenceError.memoryAllocation(
    String field0,
  ) = InferenceError_MemoryAllocation;
  const factory InferenceError.threadPool(
    String field0,
  ) = InferenceError_ThreadPool;
  const factory InferenceError.gpu(
    String field0,
  ) = InferenceError_Gpu;
  const factory InferenceError.formatDetection(
    String field0,
  ) = InferenceError_FormatDetection;
}
