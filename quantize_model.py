import tensorflow as tf
import os

def quantize_model():
    model_path = "best_model.keras"
    output_path = "best_model.tflite"

    print(f"Loading Keras model from {model_path}...")
    try:
        model = tf.keras.models.load_model(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Converting model to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Set the optimization flag for quantization
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Specify that we want Float16 quantization
    converter.target_spec.supported_types = [tf.float16]
    
    try:
        tflite_quant_model = converter.convert()
    except Exception as e:
        print(f"Error converting model: {e}")
        return
    
    print(f"Saving quantized TFLite model to {output_path}...")
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)
        
    print("Done! Here are the sizes:")
    keras_size = os.path.getsize(model_path) / (1024 * 1024)
    tflite_size = os.path.getsize(output_path) / (1024 * 1024)
    print(f"Original size: {keras_size:.2f} MB")
    print(f"Quantized size: {tflite_size:.2f} MB")

if __name__ == "__main__":
    quantize_model()
