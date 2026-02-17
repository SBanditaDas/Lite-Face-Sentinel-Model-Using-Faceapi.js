/**
 * Model Configuration for Face Recognition
 * Centralized settings for the face recognition model
 */

export const MODEL_CONFIG = {
  // Model paths
  MODEL_PATH: '/models/face_recognition/model.json',
  BLAZEFACE_MODEL: '@tensorflow-models/blazeface',

  // Input specifications
  INPUT_SIZE: 112, // MobileFaceNet expects 112x112 images
  EMBEDDING_SIZE: 128, // Output embedding dimension

  // Image preprocessing
  MEAN_RGB: [127.5, 127.5, 127.5],
  STD_RGB: [128.0, 128.0, 128.0],

  // Face verification thresholds (Optimal balance for production)
  SIMILARITY_THRESHOLD: 0.78, // Adjusted to 78% for reliable user recognition
  HIGH_CONFIDENCE_THRESHOLD: 0.84, // High confidence match

  // Performance settings
  BACKEND: 'webgl', // Use WebGL for GPU acceleration
  MAX_FACES: 1, // Process only the largest/closest face

  // Real-time verification
  VERIFICATION_INTERVAL_MS: 500, // Check every 500ms

  // Face detection settings (BlazeFace)
  FACE_DETECTION_CONFIG: {
    maxFaces: 1,
    iouThreshold: 0.3,
    scoreThreshold: 0.75,
  },
};

export default MODEL_CONFIG;
