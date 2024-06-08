import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

// Main Function
void main() {
  runApp(const MyApp());
}

// Basic Structure of App
class MyApp extends StatelessWidget {
  const MyApp({super.key});

// Overall Structure and Theme
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

//Represents Main Screen of the APP
class MyHomePage extends StatefulWidget {
  const MyHomePage({super.key});

  @override
  _MyHomePageState createState() => _MyHomePageState();
}

//State Class for MyHomePage
class _MyHomePageState extends State<MyHomePage> {
  File? _image; // Holds the selected image
  late Uint8List _imageData; //Object holds the image data
  List<String> _outputLabels =
      []; // List of strings that holds classification results
  late Interpreter
      _interpreter; // Tflite interpreter used for running the model

  // Initialize the model
  @override
  void initState() {
    super.initState();
    loadModel();
  }

  // Loads the Tflite model from assets folder
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/unet_model.tflite');
    } catch (e) {
      if (kDebugMode) {
        print('Failed to load model: $e');
      }
    }
  }

  //Use to pick the image from gallery
  Future<void> getImage() async {
    final pickedFile =
        await ImagePicker().pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _imageData = _image!.readAsBytesSync();
      });
      classifyImage();
    }
  }

  //Run the image classification
  Future<void> classifyImage() async {
    if (_imageData.isEmpty) return;

    // Perform inference
    var input = _imageData.buffer.asUint8List();
    var output =
        List<double>.filled(_interpreter.getOutputTensor(0).shape[1], 0);
    _interpreter.run(input, output);

    // Get labels
    var labels = await File('assets/labels.txt').readAsString();
    var outputLabels = labels.split('\n');

    setState(() {
      _outputLabels = outputLabels;
    });
  }

  // Build the widget tree for MyHomePage screen
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'Backlit Image Classification',
          style: TextStyle(color: Colors.white),
        ),
        centerTitle: true,
        backgroundColor: Colors.blueGrey[900],
      ),
      body: Center(
        child: _image == null
            ? const Text('No image selected, please select')
            : Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: <Widget>[
                  Image.file(_image!),
                  const SizedBox(height: 20),
                  const Text('Please Wait! Loading Classification Results:'),
                  const SizedBox(height: 10),
                  _outputLabels.isEmpty
                      ? const CircularProgressIndicator()
                      : Column(
                          children: _outputLabels
                              .map((label) => Text(label))
                              .toList(),
                        ),
                ],
              ),
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: getImage,
        tooltip: 'Pick Image',
        child: const Icon(Icons.add_a_photo),
      ),
    );
  }
}
