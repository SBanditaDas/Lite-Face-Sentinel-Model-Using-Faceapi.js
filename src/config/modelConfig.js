/**
 * Model Configuration for Face Recognition (High Accuracy)
 */

export const MODEL_CONFIG = {
  // Euclidean distance thresholds (Lower = More Similar)
  // 0.6 is the industry standard for face-api.js descriptors
  SIMILARITY_THRESHOLD: 0.60,
  HIGH_CONFIDENCE: 0.45,
  VERY_HIGH_CONFIDENCE: 0.35,

  // Verification timing
  VERIFICATION_INTERVAL_MS: 500,

  // Model assets
  MODEL_URL: '/models',
};

export default MODEL_CONFIG;
