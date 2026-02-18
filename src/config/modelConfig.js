/**
 * Model Configuration for Face Recognition
 */

export const MODEL_CONFIG = {
  MODEL_PATH: '/models/face_recognition/model.json',
  BLAZEFACE_MODEL: '@tensorflow-models/blazeface',
  INPUT_SIZE: 112,
  EMBEDDING_SIZE: 128,
  MEAN_RGB: [127.5, 127.5, 127.5],
  STD_RGB: [128.0, 128.0, 128.0],
  SIMILARITY_THRESHOLD: 0.78,
  HIGH_CONFIDENCE_THRESHOLD: 0.84,
  BACKEND: 'webgl',
  MAX_FACES: 1,
  VERIFICATION_INTERVAL_MS: 500,
  FACE_DETECTION_CONFIG: {
    maxFaces: 1,
    iouThreshold: 0.3,
    scoreThreshold: 0.75,
  },
};

export default MODEL_CONFIG;
