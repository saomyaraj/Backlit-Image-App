import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter_plus/tflite_flutter_plus.dart'; // Use tflite_flutter_plus instead of tflite_flutter
import 'package:tflite_flutter_helper_plus/tflite_flutter_helper_plus.dart'; // Ensure helper package matches
import 'package:image/image.dart' as img;

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'TFLite Image Classification',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: const MyHomePage(),
    );
  }
}

class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

class _MyHomePageState extends State<MyHomePage> {
  File? _image;
  String? _outputMessage;
  late Interpreter _interpreter;
  late TensorImage _inputImage;
  late TensorBuffer _outputBuffer;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('gen3_quantized.tflite');
      var inputShape = _interpreter.getInputTensor(0).shape;
      var outputShape = _interpreter.getOutputTensor(0).shape;

      _inputImage = TensorImage(
          TfLiteType.uint8); // Use TfLiteType from tflite_flutter_plus
      _outputBuffer = TensorBuffer.createFixedSize(outputShape,
          TfLiteType.float32); // Use TfLiteType from tflite_flutter_plus
    } catch (e) {
      if (kDebugMode) {
        print('Error loading model: $e');
      }
    }
  }

  void _preprocessImage(img.Image image) {
    // Resize the image to the input size expected by the model
    var resizedImage = img.copyResize(image,
        width: _inputImage.width, height: _inputImage.height);

    // Load the image into TensorImage
    _inputImage.loadImage(resizedImage);
  }

  Future<void> _classifyImage(File image) async {
    try {
      final bytes = await image.readAsBytes();
      final imageLib = img.decodeImage(bytes);

      // Preprocess the image
      _preprocessImage(imageLib!);

      _interpreter.run(_inputImage.buffer, _outputBuffer.buffer);

      setState(() {
        _outputMessage = 'Output: ${_outputBuffer.getDoubleList()}';
      });
    } catch (e) {
      setState(() {
        _outputMessage = 'Error classifying image: $e';
      });
    }
  }

  Future<void> _getImage() async {
    final pickedFile =
        await ImagePicker().pickImage(source: ImageSource.gallery);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
      });
      await _classifyImage(_image!);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Image Classification'),
        centerTitle: true,
        backgroundColor: Colors.blueGrey[900],
      ),
      body: Center(
        child: _image == null
            ? const Text('No image selected, please select an image.')
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Image.file(_image!),
                  const SizedBox(height: 20),
                  _outputMessage == null
                      ? const CircularProgressIndicator()
                      : Text(_outputMessage!),
                ],
              ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: _getImage,
        tooltip: 'Pick Image',
        child: const Icon(Icons.add_a_photo),
      ),
    );
  }
}
