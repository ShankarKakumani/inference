// GENERATED CODE - DO NOT MODIFY BY HAND
// coverage:ignore-file
// ignore_for_file: type=lint
// ignore_for_file: unused_element, deprecated_member_use, deprecated_member_use_from_same_package, use_function_type_syntax_for_parameters, unnecessary_const, avoid_init_to_null, invalid_override_different_default_values_named, prefer_expression_function_bodies, annotate_overrides, invalid_annotation_target, unnecessary_question_mark

part of 'error.dart';

// **************************************************************************
// FreezedGenerator
// **************************************************************************

// dart format off
T _$identity<T>(T value) => value;

/// @nodoc
mixin _$InferenceError {
  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType && other is InferenceError);
  }

  @override
  int get hashCode => runtimeType.hashCode;

  @override
  String toString() {
    return 'InferenceError()';
  }
}

/// @nodoc
class $InferenceErrorCopyWith<$Res> {
  $InferenceErrorCopyWith(InferenceError _, $Res Function(InferenceError) __);
}

/// Adds pattern-matching-related methods to [InferenceError].
extension InferenceErrorPatterns on InferenceError {
  /// A variant of `map` that fallback to returning `orElse`.
  ///
  /// It is equivalent to doing:
  /// ```dart
  /// switch (sealedClass) {
  ///   case final Subclass value:
  ///     return ...;
  ///   case _:
  ///     return orElse();
  /// }
  /// ```

  @optionalTypeArgs
  TResult maybeMap<TResult extends Object?>({
    TResult Function(InferenceError_ModelLoad value)? modelLoad,
    TResult Function(InferenceError_Prediction value)? prediction,
    TResult Function(InferenceError_UnsupportedFormat value)? unsupportedFormat,
    TResult Function(InferenceError_InvalidShape value)? invalidShape,
    TResult Function(InferenceError_InvalidTensorData value)? invalidTensorData,
    TResult Function(InferenceError_Engine value)? engine,
    TResult Function(InferenceError_Io value)? io,
    TResult Function(InferenceError_Serialization value)? serialization,
    TResult Function(InferenceError_Configuration value)? configuration,
    TResult Function(InferenceError_ResourceNotFound value)? resourceNotFound,
    TResult Function(InferenceError_MemoryAllocation value)? memoryAllocation,
    TResult Function(InferenceError_ThreadPool value)? threadPool,
    TResult Function(InferenceError_Gpu value)? gpu,
    TResult Function(InferenceError_FormatDetection value)? formatDetection,
    required TResult orElse(),
  }) {
    final _that = this;
    switch (_that) {
      case InferenceError_ModelLoad() when modelLoad != null:
        return modelLoad(_that);
      case InferenceError_Prediction() when prediction != null:
        return prediction(_that);
      case InferenceError_UnsupportedFormat() when unsupportedFormat != null:
        return unsupportedFormat(_that);
      case InferenceError_InvalidShape() when invalidShape != null:
        return invalidShape(_that);
      case InferenceError_InvalidTensorData() when invalidTensorData != null:
        return invalidTensorData(_that);
      case InferenceError_Engine() when engine != null:
        return engine(_that);
      case InferenceError_Io() when io != null:
        return io(_that);
      case InferenceError_Serialization() when serialization != null:
        return serialization(_that);
      case InferenceError_Configuration() when configuration != null:
        return configuration(_that);
      case InferenceError_ResourceNotFound() when resourceNotFound != null:
        return resourceNotFound(_that);
      case InferenceError_MemoryAllocation() when memoryAllocation != null:
        return memoryAllocation(_that);
      case InferenceError_ThreadPool() when threadPool != null:
        return threadPool(_that);
      case InferenceError_Gpu() when gpu != null:
        return gpu(_that);
      case InferenceError_FormatDetection() when formatDetection != null:
        return formatDetection(_that);
      case _:
        return orElse();
    }
  }

  /// A `switch`-like method, using callbacks.
  ///
  /// Callbacks receives the raw object, upcasted.
  /// It is equivalent to doing:
  /// ```dart
  /// switch (sealedClass) {
  ///   case final Subclass value:
  ///     return ...;
  ///   case final Subclass2 value:
  ///     return ...;
  /// }
  /// ```

  @optionalTypeArgs
  TResult map<TResult extends Object?>({
    required TResult Function(InferenceError_ModelLoad value) modelLoad,
    required TResult Function(InferenceError_Prediction value) prediction,
    required TResult Function(InferenceError_UnsupportedFormat value)
        unsupportedFormat,
    required TResult Function(InferenceError_InvalidShape value) invalidShape,
    required TResult Function(InferenceError_InvalidTensorData value)
        invalidTensorData,
    required TResult Function(InferenceError_Engine value) engine,
    required TResult Function(InferenceError_Io value) io,
    required TResult Function(InferenceError_Serialization value) serialization,
    required TResult Function(InferenceError_Configuration value) configuration,
    required TResult Function(InferenceError_ResourceNotFound value)
        resourceNotFound,
    required TResult Function(InferenceError_MemoryAllocation value)
        memoryAllocation,
    required TResult Function(InferenceError_ThreadPool value) threadPool,
    required TResult Function(InferenceError_Gpu value) gpu,
    required TResult Function(InferenceError_FormatDetection value)
        formatDetection,
  }) {
    final _that = this;
    switch (_that) {
      case InferenceError_ModelLoad():
        return modelLoad(_that);
      case InferenceError_Prediction():
        return prediction(_that);
      case InferenceError_UnsupportedFormat():
        return unsupportedFormat(_that);
      case InferenceError_InvalidShape():
        return invalidShape(_that);
      case InferenceError_InvalidTensorData():
        return invalidTensorData(_that);
      case InferenceError_Engine():
        return engine(_that);
      case InferenceError_Io():
        return io(_that);
      case InferenceError_Serialization():
        return serialization(_that);
      case InferenceError_Configuration():
        return configuration(_that);
      case InferenceError_ResourceNotFound():
        return resourceNotFound(_that);
      case InferenceError_MemoryAllocation():
        return memoryAllocation(_that);
      case InferenceError_ThreadPool():
        return threadPool(_that);
      case InferenceError_Gpu():
        return gpu(_that);
      case InferenceError_FormatDetection():
        return formatDetection(_that);
    }
  }

  /// A variant of `map` that fallback to returning `null`.
  ///
  /// It is equivalent to doing:
  /// ```dart
  /// switch (sealedClass) {
  ///   case final Subclass value:
  ///     return ...;
  ///   case _:
  ///     return null;
  /// }
  /// ```

  @optionalTypeArgs
  TResult? mapOrNull<TResult extends Object?>({
    TResult? Function(InferenceError_ModelLoad value)? modelLoad,
    TResult? Function(InferenceError_Prediction value)? prediction,
    TResult? Function(InferenceError_UnsupportedFormat value)?
        unsupportedFormat,
    TResult? Function(InferenceError_InvalidShape value)? invalidShape,
    TResult? Function(InferenceError_InvalidTensorData value)?
        invalidTensorData,
    TResult? Function(InferenceError_Engine value)? engine,
    TResult? Function(InferenceError_Io value)? io,
    TResult? Function(InferenceError_Serialization value)? serialization,
    TResult? Function(InferenceError_Configuration value)? configuration,
    TResult? Function(InferenceError_ResourceNotFound value)? resourceNotFound,
    TResult? Function(InferenceError_MemoryAllocation value)? memoryAllocation,
    TResult? Function(InferenceError_ThreadPool value)? threadPool,
    TResult? Function(InferenceError_Gpu value)? gpu,
    TResult? Function(InferenceError_FormatDetection value)? formatDetection,
  }) {
    final _that = this;
    switch (_that) {
      case InferenceError_ModelLoad() when modelLoad != null:
        return modelLoad(_that);
      case InferenceError_Prediction() when prediction != null:
        return prediction(_that);
      case InferenceError_UnsupportedFormat() when unsupportedFormat != null:
        return unsupportedFormat(_that);
      case InferenceError_InvalidShape() when invalidShape != null:
        return invalidShape(_that);
      case InferenceError_InvalidTensorData() when invalidTensorData != null:
        return invalidTensorData(_that);
      case InferenceError_Engine() when engine != null:
        return engine(_that);
      case InferenceError_Io() when io != null:
        return io(_that);
      case InferenceError_Serialization() when serialization != null:
        return serialization(_that);
      case InferenceError_Configuration() when configuration != null:
        return configuration(_that);
      case InferenceError_ResourceNotFound() when resourceNotFound != null:
        return resourceNotFound(_that);
      case InferenceError_MemoryAllocation() when memoryAllocation != null:
        return memoryAllocation(_that);
      case InferenceError_ThreadPool() when threadPool != null:
        return threadPool(_that);
      case InferenceError_Gpu() when gpu != null:
        return gpu(_that);
      case InferenceError_FormatDetection() when formatDetection != null:
        return formatDetection(_that);
      case _:
        return null;
    }
  }

  /// A variant of `when` that fallback to an `orElse` callback.
  ///
  /// It is equivalent to doing:
  /// ```dart
  /// switch (sealedClass) {
  ///   case Subclass(:final field):
  ///     return ...;
  ///   case _:
  ///     return orElse();
  /// }
  /// ```

  @optionalTypeArgs
  TResult maybeWhen<TResult extends Object?>({
    TResult Function(String field0)? modelLoad,
    TResult Function(String field0)? prediction,
    TResult Function(String field0)? unsupportedFormat,
    TResult Function(Uint64List expected, Uint64List actual)? invalidShape,
    TResult Function(String field0)? invalidTensorData,
    TResult Function(String field0)? engine,
    TResult Function(String field0)? io,
    TResult Function(String field0)? serialization,
    TResult Function(String field0)? configuration,
    TResult Function(String field0)? resourceNotFound,
    TResult Function(String field0)? memoryAllocation,
    TResult Function(String field0)? threadPool,
    TResult Function(String field0)? gpu,
    TResult Function(String field0)? formatDetection,
    required TResult orElse(),
  }) {
    final _that = this;
    switch (_that) {
      case InferenceError_ModelLoad() when modelLoad != null:
        return modelLoad(_that.field0);
      case InferenceError_Prediction() when prediction != null:
        return prediction(_that.field0);
      case InferenceError_UnsupportedFormat() when unsupportedFormat != null:
        return unsupportedFormat(_that.field0);
      case InferenceError_InvalidShape() when invalidShape != null:
        return invalidShape(_that.expected, _that.actual);
      case InferenceError_InvalidTensorData() when invalidTensorData != null:
        return invalidTensorData(_that.field0);
      case InferenceError_Engine() when engine != null:
        return engine(_that.field0);
      case InferenceError_Io() when io != null:
        return io(_that.field0);
      case InferenceError_Serialization() when serialization != null:
        return serialization(_that.field0);
      case InferenceError_Configuration() when configuration != null:
        return configuration(_that.field0);
      case InferenceError_ResourceNotFound() when resourceNotFound != null:
        return resourceNotFound(_that.field0);
      case InferenceError_MemoryAllocation() when memoryAllocation != null:
        return memoryAllocation(_that.field0);
      case InferenceError_ThreadPool() when threadPool != null:
        return threadPool(_that.field0);
      case InferenceError_Gpu() when gpu != null:
        return gpu(_that.field0);
      case InferenceError_FormatDetection() when formatDetection != null:
        return formatDetection(_that.field0);
      case _:
        return orElse();
    }
  }

  /// A `switch`-like method, using callbacks.
  ///
  /// As opposed to `map`, this offers destructuring.
  /// It is equivalent to doing:
  /// ```dart
  /// switch (sealedClass) {
  ///   case Subclass(:final field):
  ///     return ...;
  ///   case Subclass2(:final field2):
  ///     return ...;
  /// }
  /// ```

  @optionalTypeArgs
  TResult when<TResult extends Object?>({
    required TResult Function(String field0) modelLoad,
    required TResult Function(String field0) prediction,
    required TResult Function(String field0) unsupportedFormat,
    required TResult Function(Uint64List expected, Uint64List actual)
        invalidShape,
    required TResult Function(String field0) invalidTensorData,
    required TResult Function(String field0) engine,
    required TResult Function(String field0) io,
    required TResult Function(String field0) serialization,
    required TResult Function(String field0) configuration,
    required TResult Function(String field0) resourceNotFound,
    required TResult Function(String field0) memoryAllocation,
    required TResult Function(String field0) threadPool,
    required TResult Function(String field0) gpu,
    required TResult Function(String field0) formatDetection,
  }) {
    final _that = this;
    switch (_that) {
      case InferenceError_ModelLoad():
        return modelLoad(_that.field0);
      case InferenceError_Prediction():
        return prediction(_that.field0);
      case InferenceError_UnsupportedFormat():
        return unsupportedFormat(_that.field0);
      case InferenceError_InvalidShape():
        return invalidShape(_that.expected, _that.actual);
      case InferenceError_InvalidTensorData():
        return invalidTensorData(_that.field0);
      case InferenceError_Engine():
        return engine(_that.field0);
      case InferenceError_Io():
        return io(_that.field0);
      case InferenceError_Serialization():
        return serialization(_that.field0);
      case InferenceError_Configuration():
        return configuration(_that.field0);
      case InferenceError_ResourceNotFound():
        return resourceNotFound(_that.field0);
      case InferenceError_MemoryAllocation():
        return memoryAllocation(_that.field0);
      case InferenceError_ThreadPool():
        return threadPool(_that.field0);
      case InferenceError_Gpu():
        return gpu(_that.field0);
      case InferenceError_FormatDetection():
        return formatDetection(_that.field0);
    }
  }

  /// A variant of `when` that fallback to returning `null`
  ///
  /// It is equivalent to doing:
  /// ```dart
  /// switch (sealedClass) {
  ///   case Subclass(:final field):
  ///     return ...;
  ///   case _:
  ///     return null;
  /// }
  /// ```

  @optionalTypeArgs
  TResult? whenOrNull<TResult extends Object?>({
    TResult? Function(String field0)? modelLoad,
    TResult? Function(String field0)? prediction,
    TResult? Function(String field0)? unsupportedFormat,
    TResult? Function(Uint64List expected, Uint64List actual)? invalidShape,
    TResult? Function(String field0)? invalidTensorData,
    TResult? Function(String field0)? engine,
    TResult? Function(String field0)? io,
    TResult? Function(String field0)? serialization,
    TResult? Function(String field0)? configuration,
    TResult? Function(String field0)? resourceNotFound,
    TResult? Function(String field0)? memoryAllocation,
    TResult? Function(String field0)? threadPool,
    TResult? Function(String field0)? gpu,
    TResult? Function(String field0)? formatDetection,
  }) {
    final _that = this;
    switch (_that) {
      case InferenceError_ModelLoad() when modelLoad != null:
        return modelLoad(_that.field0);
      case InferenceError_Prediction() when prediction != null:
        return prediction(_that.field0);
      case InferenceError_UnsupportedFormat() when unsupportedFormat != null:
        return unsupportedFormat(_that.field0);
      case InferenceError_InvalidShape() when invalidShape != null:
        return invalidShape(_that.expected, _that.actual);
      case InferenceError_InvalidTensorData() when invalidTensorData != null:
        return invalidTensorData(_that.field0);
      case InferenceError_Engine() when engine != null:
        return engine(_that.field0);
      case InferenceError_Io() when io != null:
        return io(_that.field0);
      case InferenceError_Serialization() when serialization != null:
        return serialization(_that.field0);
      case InferenceError_Configuration() when configuration != null:
        return configuration(_that.field0);
      case InferenceError_ResourceNotFound() when resourceNotFound != null:
        return resourceNotFound(_that.field0);
      case InferenceError_MemoryAllocation() when memoryAllocation != null:
        return memoryAllocation(_that.field0);
      case InferenceError_ThreadPool() when threadPool != null:
        return threadPool(_that.field0);
      case InferenceError_Gpu() when gpu != null:
        return gpu(_that.field0);
      case InferenceError_FormatDetection() when formatDetection != null:
        return formatDetection(_that.field0);
      case _:
        return null;
    }
  }
}

/// @nodoc

class InferenceError_ModelLoad extends InferenceError {
  const InferenceError_ModelLoad(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_ModelLoadCopyWith<InferenceError_ModelLoad> get copyWith =>
      _$InferenceError_ModelLoadCopyWithImpl<InferenceError_ModelLoad>(
          this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_ModelLoad &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.modelLoad(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_ModelLoadCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_ModelLoadCopyWith(InferenceError_ModelLoad value,
          $Res Function(InferenceError_ModelLoad) _then) =
      _$InferenceError_ModelLoadCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_ModelLoadCopyWithImpl<$Res>
    implements $InferenceError_ModelLoadCopyWith<$Res> {
  _$InferenceError_ModelLoadCopyWithImpl(this._self, this._then);

  final InferenceError_ModelLoad _self;
  final $Res Function(InferenceError_ModelLoad) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_ModelLoad(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_Prediction extends InferenceError {
  const InferenceError_Prediction(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_PredictionCopyWith<InferenceError_Prediction> get copyWith =>
      _$InferenceError_PredictionCopyWithImpl<InferenceError_Prediction>(
          this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_Prediction &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.prediction(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_PredictionCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_PredictionCopyWith(InferenceError_Prediction value,
          $Res Function(InferenceError_Prediction) _then) =
      _$InferenceError_PredictionCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_PredictionCopyWithImpl<$Res>
    implements $InferenceError_PredictionCopyWith<$Res> {
  _$InferenceError_PredictionCopyWithImpl(this._self, this._then);

  final InferenceError_Prediction _self;
  final $Res Function(InferenceError_Prediction) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_Prediction(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_UnsupportedFormat extends InferenceError {
  const InferenceError_UnsupportedFormat(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_UnsupportedFormatCopyWith<InferenceError_UnsupportedFormat>
      get copyWith => _$InferenceError_UnsupportedFormatCopyWithImpl<
          InferenceError_UnsupportedFormat>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_UnsupportedFormat &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.unsupportedFormat(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_UnsupportedFormatCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_UnsupportedFormatCopyWith(
          InferenceError_UnsupportedFormat value,
          $Res Function(InferenceError_UnsupportedFormat) _then) =
      _$InferenceError_UnsupportedFormatCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_UnsupportedFormatCopyWithImpl<$Res>
    implements $InferenceError_UnsupportedFormatCopyWith<$Res> {
  _$InferenceError_UnsupportedFormatCopyWithImpl(this._self, this._then);

  final InferenceError_UnsupportedFormat _self;
  final $Res Function(InferenceError_UnsupportedFormat) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_UnsupportedFormat(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_InvalidShape extends InferenceError {
  const InferenceError_InvalidShape(
      {required this.expected, required this.actual})
      : super._();

  final Uint64List expected;
  final Uint64List actual;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_InvalidShapeCopyWith<InferenceError_InvalidShape>
      get copyWith => _$InferenceError_InvalidShapeCopyWithImpl<
          InferenceError_InvalidShape>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_InvalidShape &&
            const DeepCollectionEquality().equals(other.expected, expected) &&
            const DeepCollectionEquality().equals(other.actual, actual));
  }

  @override
  int get hashCode => Object.hash(
      runtimeType,
      const DeepCollectionEquality().hash(expected),
      const DeepCollectionEquality().hash(actual));

  @override
  String toString() {
    return 'InferenceError.invalidShape(expected: $expected, actual: $actual)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_InvalidShapeCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_InvalidShapeCopyWith(
          InferenceError_InvalidShape value,
          $Res Function(InferenceError_InvalidShape) _then) =
      _$InferenceError_InvalidShapeCopyWithImpl;
  @useResult
  $Res call({Uint64List expected, Uint64List actual});
}

/// @nodoc
class _$InferenceError_InvalidShapeCopyWithImpl<$Res>
    implements $InferenceError_InvalidShapeCopyWith<$Res> {
  _$InferenceError_InvalidShapeCopyWithImpl(this._self, this._then);

  final InferenceError_InvalidShape _self;
  final $Res Function(InferenceError_InvalidShape) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? expected = null,
    Object? actual = null,
  }) {
    return _then(InferenceError_InvalidShape(
      expected: null == expected
          ? _self.expected
          : expected // ignore: cast_nullable_to_non_nullable
              as Uint64List,
      actual: null == actual
          ? _self.actual
          : actual // ignore: cast_nullable_to_non_nullable
              as Uint64List,
    ));
  }
}

/// @nodoc

class InferenceError_InvalidTensorData extends InferenceError {
  const InferenceError_InvalidTensorData(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_InvalidTensorDataCopyWith<InferenceError_InvalidTensorData>
      get copyWith => _$InferenceError_InvalidTensorDataCopyWithImpl<
          InferenceError_InvalidTensorData>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_InvalidTensorData &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.invalidTensorData(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_InvalidTensorDataCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_InvalidTensorDataCopyWith(
          InferenceError_InvalidTensorData value,
          $Res Function(InferenceError_InvalidTensorData) _then) =
      _$InferenceError_InvalidTensorDataCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_InvalidTensorDataCopyWithImpl<$Res>
    implements $InferenceError_InvalidTensorDataCopyWith<$Res> {
  _$InferenceError_InvalidTensorDataCopyWithImpl(this._self, this._then);

  final InferenceError_InvalidTensorData _self;
  final $Res Function(InferenceError_InvalidTensorData) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_InvalidTensorData(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_Engine extends InferenceError {
  const InferenceError_Engine(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_EngineCopyWith<InferenceError_Engine> get copyWith =>
      _$InferenceError_EngineCopyWithImpl<InferenceError_Engine>(
          this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_Engine &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.engine(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_EngineCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_EngineCopyWith(InferenceError_Engine value,
          $Res Function(InferenceError_Engine) _then) =
      _$InferenceError_EngineCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_EngineCopyWithImpl<$Res>
    implements $InferenceError_EngineCopyWith<$Res> {
  _$InferenceError_EngineCopyWithImpl(this._self, this._then);

  final InferenceError_Engine _self;
  final $Res Function(InferenceError_Engine) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_Engine(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_Io extends InferenceError {
  const InferenceError_Io(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_IoCopyWith<InferenceError_Io> get copyWith =>
      _$InferenceError_IoCopyWithImpl<InferenceError_Io>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_Io &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.io(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_IoCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_IoCopyWith(
          InferenceError_Io value, $Res Function(InferenceError_Io) _then) =
      _$InferenceError_IoCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_IoCopyWithImpl<$Res>
    implements $InferenceError_IoCopyWith<$Res> {
  _$InferenceError_IoCopyWithImpl(this._self, this._then);

  final InferenceError_Io _self;
  final $Res Function(InferenceError_Io) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_Io(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_Serialization extends InferenceError {
  const InferenceError_Serialization(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_SerializationCopyWith<InferenceError_Serialization>
      get copyWith => _$InferenceError_SerializationCopyWithImpl<
          InferenceError_Serialization>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_Serialization &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.serialization(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_SerializationCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_SerializationCopyWith(
          InferenceError_Serialization value,
          $Res Function(InferenceError_Serialization) _then) =
      _$InferenceError_SerializationCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_SerializationCopyWithImpl<$Res>
    implements $InferenceError_SerializationCopyWith<$Res> {
  _$InferenceError_SerializationCopyWithImpl(this._self, this._then);

  final InferenceError_Serialization _self;
  final $Res Function(InferenceError_Serialization) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_Serialization(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_Configuration extends InferenceError {
  const InferenceError_Configuration(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_ConfigurationCopyWith<InferenceError_Configuration>
      get copyWith => _$InferenceError_ConfigurationCopyWithImpl<
          InferenceError_Configuration>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_Configuration &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.configuration(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_ConfigurationCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_ConfigurationCopyWith(
          InferenceError_Configuration value,
          $Res Function(InferenceError_Configuration) _then) =
      _$InferenceError_ConfigurationCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_ConfigurationCopyWithImpl<$Res>
    implements $InferenceError_ConfigurationCopyWith<$Res> {
  _$InferenceError_ConfigurationCopyWithImpl(this._self, this._then);

  final InferenceError_Configuration _self;
  final $Res Function(InferenceError_Configuration) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_Configuration(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_ResourceNotFound extends InferenceError {
  const InferenceError_ResourceNotFound(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_ResourceNotFoundCopyWith<InferenceError_ResourceNotFound>
      get copyWith => _$InferenceError_ResourceNotFoundCopyWithImpl<
          InferenceError_ResourceNotFound>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_ResourceNotFound &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.resourceNotFound(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_ResourceNotFoundCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_ResourceNotFoundCopyWith(
          InferenceError_ResourceNotFound value,
          $Res Function(InferenceError_ResourceNotFound) _then) =
      _$InferenceError_ResourceNotFoundCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_ResourceNotFoundCopyWithImpl<$Res>
    implements $InferenceError_ResourceNotFoundCopyWith<$Res> {
  _$InferenceError_ResourceNotFoundCopyWithImpl(this._self, this._then);

  final InferenceError_ResourceNotFound _self;
  final $Res Function(InferenceError_ResourceNotFound) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_ResourceNotFound(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_MemoryAllocation extends InferenceError {
  const InferenceError_MemoryAllocation(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_MemoryAllocationCopyWith<InferenceError_MemoryAllocation>
      get copyWith => _$InferenceError_MemoryAllocationCopyWithImpl<
          InferenceError_MemoryAllocation>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_MemoryAllocation &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.memoryAllocation(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_MemoryAllocationCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_MemoryAllocationCopyWith(
          InferenceError_MemoryAllocation value,
          $Res Function(InferenceError_MemoryAllocation) _then) =
      _$InferenceError_MemoryAllocationCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_MemoryAllocationCopyWithImpl<$Res>
    implements $InferenceError_MemoryAllocationCopyWith<$Res> {
  _$InferenceError_MemoryAllocationCopyWithImpl(this._self, this._then);

  final InferenceError_MemoryAllocation _self;
  final $Res Function(InferenceError_MemoryAllocation) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_MemoryAllocation(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_ThreadPool extends InferenceError {
  const InferenceError_ThreadPool(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_ThreadPoolCopyWith<InferenceError_ThreadPool> get copyWith =>
      _$InferenceError_ThreadPoolCopyWithImpl<InferenceError_ThreadPool>(
          this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_ThreadPool &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.threadPool(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_ThreadPoolCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_ThreadPoolCopyWith(InferenceError_ThreadPool value,
          $Res Function(InferenceError_ThreadPool) _then) =
      _$InferenceError_ThreadPoolCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_ThreadPoolCopyWithImpl<$Res>
    implements $InferenceError_ThreadPoolCopyWith<$Res> {
  _$InferenceError_ThreadPoolCopyWithImpl(this._self, this._then);

  final InferenceError_ThreadPool _self;
  final $Res Function(InferenceError_ThreadPool) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_ThreadPool(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_Gpu extends InferenceError {
  const InferenceError_Gpu(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_GpuCopyWith<InferenceError_Gpu> get copyWith =>
      _$InferenceError_GpuCopyWithImpl<InferenceError_Gpu>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_Gpu &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.gpu(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_GpuCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_GpuCopyWith(
          InferenceError_Gpu value, $Res Function(InferenceError_Gpu) _then) =
      _$InferenceError_GpuCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_GpuCopyWithImpl<$Res>
    implements $InferenceError_GpuCopyWith<$Res> {
  _$InferenceError_GpuCopyWithImpl(this._self, this._then);

  final InferenceError_Gpu _self;
  final $Res Function(InferenceError_Gpu) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_Gpu(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

/// @nodoc

class InferenceError_FormatDetection extends InferenceError {
  const InferenceError_FormatDetection(this.field0) : super._();

  final String field0;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @JsonKey(includeFromJson: false, includeToJson: false)
  @pragma('vm:prefer-inline')
  $InferenceError_FormatDetectionCopyWith<InferenceError_FormatDetection>
      get copyWith => _$InferenceError_FormatDetectionCopyWithImpl<
          InferenceError_FormatDetection>(this, _$identity);

  @override
  bool operator ==(Object other) {
    return identical(this, other) ||
        (other.runtimeType == runtimeType &&
            other is InferenceError_FormatDetection &&
            (identical(other.field0, field0) || other.field0 == field0));
  }

  @override
  int get hashCode => Object.hash(runtimeType, field0);

  @override
  String toString() {
    return 'InferenceError.formatDetection(field0: $field0)';
  }
}

/// @nodoc
abstract mixin class $InferenceError_FormatDetectionCopyWith<$Res>
    implements $InferenceErrorCopyWith<$Res> {
  factory $InferenceError_FormatDetectionCopyWith(
          InferenceError_FormatDetection value,
          $Res Function(InferenceError_FormatDetection) _then) =
      _$InferenceError_FormatDetectionCopyWithImpl;
  @useResult
  $Res call({String field0});
}

/// @nodoc
class _$InferenceError_FormatDetectionCopyWithImpl<$Res>
    implements $InferenceError_FormatDetectionCopyWith<$Res> {
  _$InferenceError_FormatDetectionCopyWithImpl(this._self, this._then);

  final InferenceError_FormatDetection _self;
  final $Res Function(InferenceError_FormatDetection) _then;

  /// Create a copy of InferenceError
  /// with the given fields replaced by the non-null parameter values.
  @pragma('vm:prefer-inline')
  $Res call({
    Object? field0 = null,
  }) {
    return _then(InferenceError_FormatDetection(
      null == field0
          ? _self.field0
          : field0 // ignore: cast_nullable_to_non_nullable
              as String,
    ));
  }
}

// dart format on
