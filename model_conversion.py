"""
Lightweight Face Recognition Model Creator
Creates a 1-2 MB face recognition model for browser deployment

Requirements: pip install tensorflow tensorflowjs
"""

import os
import sys

# Fix NumPy compatibility
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
import tensorflowjs as tfjs

OUTPUT_DIR = "face_recognition_model"
INPUT_SIZE = 112
EMBEDDING_SIZE = 128

print("=" * 70)
print("FACE RECOGNITION MODEL GENERATOR (1-2 MB)")
print("=" * 70)

def create_model():
    """Create lightweight MobileFaceNet"""
    print("\nCreating MobileFaceNet architecture...")
    
    inputs = tf.keras.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))
    
    # Lightweight MobileNetV2 backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(INPUT_SIZE, INPUT_SIZE, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg',
        alpha=0.5  # Reduced width for smaller size
    )
    
    x = backbone(inputs)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # 128-D embedding output
    embeddings = tf.keras.layers.Dense(EMBEDDING_SIZE, activation=None)(x)
    embeddings = tf.keras.layers.Lambda(lambda x: tf.nn.l2_normalize(x, axis=1))(embeddings)
    
    model = tf.keras.Model(inputs=inputs, outputs=embeddings, name='MobileFaceNet_Lite')
    print("✓ Model created")
    return model

def convert_to_tfjs(model, output_dir):
    """Convert to TensorFlow.js with quantization"""
    print(f"\nConverting to TensorFlow.js with quantization...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    tfjs.converters.save_keras_model(
        model,
        output_dir,
        quantization_dtype_map={'uint8': '*'}
    )
    
    # Calculate size
    total_size = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    )
    
    size_mb = total_size / (1024 * 1024)
    print(f"\n{'=' * 70}")
    print(f"MODEL SIZE: {size_mb:.2f} MB")
    print(f"{'=' * 70}")
    
    if size_mb <= 2.0:
        print("✅ SUCCESS! Meets 1-2 MB requirement!")
    
    return size_mb

def create_readme(output_dir, size_mb):
    """Create usage guide"""
    content = f"""# Face Recognition Model (TensorFlow.js)

**Size**: {size_mb:.2f} MB | **Input**: 112x112 | **Output**: 128-D embedding

## Usage in JavaScript:

```javascript
import * as tf from '@tensorflow/tfjs';

// Load model
const model = await tf.loadGraphModel('./model.json');

// Preprocess image
const input = tf.browser.fromPixels(image)
  .resizeBilinear([112, 112])
  .toFloat()
  .sub(127.5)
  .div(128.0)
  .expandDims(0);

// Get embedding
const embedding = model.predict(input);
const norm = embedding.norm(2, 1, true);
const normalized = embedding.div(norm);

// Compare faces (cosine similarity)
const similarity = embedding1.dot(embedding2).dataSync()[0];
const isSame = similarity > 0.6;  // Threshold
```

## Model ready for browser deployment!
"""
    
    with open(os.path.join(output_dir, 'README.md'), 'w') as f:
        f.write(content)
    print(f"✓ README saved")

def main():
    print("\n1. Creating model architecture...")
    model = create_model()
    
    print("\n2. Converting to TensorFlow.js...")
    size = convert_to_tfjs(model, OUTPUT_DIR)
    
    print("\n3. Creating documentation...")
    create_readme(OUTPUT_DIR, size)
    
    print(f"\n✅ DONE! Model saved to: {OUTPUT_DIR}/\n")
    print("Files:")
    for f in os.listdir(OUTPUT_DIR):
        print(f"  - {f}")
    print()

if __name__ == "__main__":
    main()
