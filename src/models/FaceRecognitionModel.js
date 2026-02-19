/**
 * Enhanced Face Recognition using face-api.js (High Accuracy)
 * Uses 128-D Deep Learning Descriptors for 99% accuracy.
 */

import * as faceapi from 'face-api.js';
import { detectLiveness } from '../utils/livenessDetection';
import { MODEL_CONFIG } from '../config/modelConfig';

class FaceApiRecognitionModel {
    constructor() {
        this.isReady = false;
        this.modelsLoaded = false;
    }

    /**
     * Load face-api.js models (Full Recognition Net)
     */
    async loadModel() {
        try {
            const url = MODEL_CONFIG.MODEL_URL;
            await Promise.all([
                faceapi.nets.tinyFaceDetector.loadFromUri(url),
                faceapi.nets.faceLandmark68TinyNet.loadFromUri(url),
                faceapi.nets.faceRecognitionNet.loadFromUri(url)
            ]);

            this.modelsLoaded = true;
            this.isReady = true;
            return true;
        } catch (error) {
            console.error('Error loading high-accuracy models:', error);
            throw error;
        }
    }

    /**
     * Extract 128-D face descriptor
     */
    async extractEmbedding(input) {
        if (!this.isReady) throw new Error('Model not loaded');

        try {
            const detection = await faceapi
                .detectSingleFace(input, new faceapi.TinyFaceDetectorOptions({
                    inputSize: 224,
                    scoreThreshold: 0.3
                }))
                .withFaceLandmarks(true)
                .withFaceDescriptor();

            if (!detection) return null;

            const livenessResult = await detectLiveness(input, detection.landmarks, detection.detection);

            return {
                embedding: Array.from(detection.descriptor),
                liveness: livenessResult
            };
        } catch (error) {
            console.error('Error extracting descriptor:', error);
            throw error;
        }
    }

    /**
     * Standard Euclidean distance comparison
     */
    compareFaces(embedding1, embedding2) {
        if (!embedding1 || !embedding2) return 1.0; // Max distance

        // Convert back to Float32Array if they are normal arrays
        const e1 = new Float32Array(embedding1);
        const e2 = new Float32Array(embedding2);

        return faceapi.euclideanDistance(e1, e2);
    }

    /**
     * Calibrated verification for 6.6MB model
     */
    isSamePerson(embedding1, embedding2, threshold = MODEL_CONFIG.SIMILARITY_THRESHOLD) {
        const distance = this.compareFaces(embedding1, embedding2);
        let level = 'low';
        let isSame = false;

        if (distance <= MODEL_CONFIG.VERY_HIGH_CONFIDENCE) {
            level = 'very_high';
            isSame = true;
        } else if (distance <= MODEL_CONFIG.HIGH_CONFIDENCE) {
            level = 'high';
            isSame = true;
        } else if (distance <= threshold) {
            level = 'medium';
            isSame = true;
        } else if (distance <= 0.8) {
            level = 'uncertain';
            isSame = false;
        } else {
            level = 'low';
            isSame = false;
        }

        return { isSame, confidence: distance, level };
    }

    dispose() {
        this.isReady = false;
        this.modelsLoaded = false;
    }
}

export default FaceApiRecognitionModel;
