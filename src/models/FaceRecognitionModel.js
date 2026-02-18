/**
 * Enhanced Face Recognition using face-api.js
 */

import * as faceapi from 'face-api.js';
import { detectLiveness } from '../utils/livenessDetection';

const MODEL_URL = '/models';

class FaceApiRecognitionModel {
    constructor() {
        this.isReady = false;
        this.modelsLoaded = false;
    }

    /**
     * Load face-api.js models (tiny versions only)
     */
    async loadModel() {
        try {
            await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
            await faceapi.nets.faceLandmark68TinyNet.loadFromUri(MODEL_URL);

            this.modelsLoaded = true;
            this.isReady = true;
            return true;
        } catch (error) {
            console.error('Error loading models:', error);
            throw error;
        }
    }

    /**
     * Extract enhanced face descriptor with multiple feature types
     */
    async extractEmbedding(input) {
        if (!this.isReady) throw new Error('Model not loaded');

        try {
            const detection = await faceapi
                .detectSingleFace(input, new faceapi.TinyFaceDetectorOptions({
                    inputSize: 224,
                    scoreThreshold: 0.3
                }))
                .withFaceLandmarks(true);

            if (!detection) return null;

            const landmarks = detection.landmarks;
            const box = detection.detection.box;

            const livenessResult = await detectLiveness(input, landmarks, detection.detection);

            // Extract multi-dimensional feature set
            const geometricFeatures = this._extractGeometricFeatures(landmarks);
            const ratioFeatures = this._extractRatioFeatures(landmarks);
            const shapeFeatures = this._extractShapeFeatures(landmarks, box);
            const symmetryFeatures = this._extractSymmetryFeatures(landmarks);

            const embedding = [
                ...geometricFeatures,
                ...ratioFeatures,
                ...shapeFeatures,
                ...symmetryFeatures
            ];

            const norm = Math.sqrt(embedding.reduce((sum, val) => sum + val * val, 0));
            const normalized = embedding.map(val => val / norm);

            return {
                embedding: normalized,
                liveness: livenessResult
            };
        } catch (error) {
            console.error('Error extracting embedding:', error);
            throw error;
        }
    }

    _extractGeometricFeatures(landmarks) {
        const positions = landmarks.positions;
        const features = [];
        const faceWidth = this._getDistance(positions[0], positions[16]);

        features.push(this._getDistance(positions[36], positions[39]) / faceWidth);
        features.push(this._getDistance(positions[42], positions[45]) / faceWidth);
        features.push(this._getDistance(positions[39], positions[42]) / faceWidth);
        features.push(this._getDistance(positions[27], positions[30]) / faceWidth);
        features.push(this._getDistance(positions[31], positions[35]) / faceWidth);
        features.push(this._getDistance(positions[48], positions[54]) / faceWidth);
        features.push(this._getDistance(positions[51], positions[57]) / faceWidth);
        features.push(this._getDistance(positions[2], positions[14]) / faceWidth);
        features.push(this._getDistance(positions[8], positions[27]) / faceWidth);

        const keyPairs = [
            [27, 33], [31, 35], [30, 33], [27, 30],
            [21, 22], [17, 21], [22, 26], [36, 45], [39, 42], [36, 39], [42, 45], [37, 41], [43, 47], [19, 37], [24, 44],
            [48, 54], [51, 57], [50, 52], [58, 56], [48, 60], [54, 64],
            [0, 16], [4, 12], [8, 30], [27, 8], [8, 57], [0, 8], [16, 8],
            [36, 48], [45, 54], [31, 48], [35, 54], [33, 51], [27, 21], [27, 22]
        ];

        keyPairs.forEach(([i, j]) => {
            features.push(this._getDistance(positions[i], positions[j]) / faceWidth);
        });

        return features;
    }

    _extractRatioFeatures(landmarks) {
        const positions = landmarks.positions;
        const features = [];

        const leftEyeWidth = this._getDistance(positions[36], positions[39]);
        const rightEyeWidth = this._getDistance(positions[42], positions[45]);
        const leftEyeHeight = this._getDistance(positions[37], positions[41]);
        const rightEyeHeight = this._getDistance(positions[43], positions[47]);

        features.push(leftEyeWidth / (leftEyeHeight + 1e-6));
        features.push(rightEyeWidth / (rightEyeHeight + 1e-6));
        features.push(leftEyeWidth / (rightEyeWidth + 1e-6));

        const faceWidth = this._getDistance(positions[0], positions[16]);
        const faceHeight = this._getDistance(positions[8], positions[27]);
        const upperFaceHeight = this._getDistance(positions[19], positions[30]);
        const lowerFaceHeight = this._getDistance(positions[30], positions[8]);

        features.push(faceWidth / (faceHeight + 1e-6));
        features.push(upperFaceHeight / (lowerFaceHeight + 1e-6));

        const noseLength = this._getDistance(positions[27], positions[33]);
        const noseWidth = this._getDistance(positions[31], positions[35]);
        features.push(noseLength / (noseWidth + 1e-6));

        const mouthWidth = this._getDistance(positions[48], positions[54]);
        const mouthHeight = this._getDistance(positions[51], positions[57]);
        features.push(mouthWidth / (mouthHeight + 1e-6));
        features.push(this._getDistance(positions[51], positions[62]) / (this._getDistance(positions[57], positions[66]) + 1e-6));

        features.push(noseLength / (mouthWidth + 1e-6));
        features.push(faceHeight / (faceWidth + 1e-6));

        return features;
    }

    _extractShapeFeatures(landmarks, box) {
        const positions = landmarks.positions;
        const features = [];

        const jawPoints = positions.slice(0, 17);
        let jawCurvature = 0;
        for (let i = 1; i < jawPoints.length - 1; i++) {
            jawCurvature += this._getAngle(jawPoints[i - 1], jawPoints[i], jawPoints[i + 1]);
        }
        features.push(jawCurvature / jawPoints.length);
        features.push(box.width / (box.height + 1e-6));

        const faceWidth = this._getDistance(positions[0], positions[16]);
        features.push(this._getDistance(positions[1], positions[31]) / faceWidth);
        features.push(this._getDistance(positions[15], positions[35]) / faceWidth);

        return features;
    }

    _extractSymmetryFeatures(landmarks) {
        const positions = landmarks.positions;
        const features = [];
        const noseTip = positions[30];

        const leftEyeCenter = this._getCentroid(positions.slice(36, 42));
        const rightEyeCenter = this._getCentroid(positions.slice(42, 48));
        features.push(this._getDistance(leftEyeCenter, noseTip) / (this._getDistance(rightEyeCenter, noseTip) + 1e-6));

        const leftMouth = this._getCentroid(positions.slice(48, 51));
        const rightMouth = this._getCentroid(positions.slice(51, 55));
        features.push(this._getDistance(leftMouth, noseTip) / (this._getDistance(rightMouth, noseTip) + 1e-6));

        const leftJaw = positions.slice(0, 9);
        const rightJaw = positions.slice(8, 17).reverse();
        let symmetryScore = 0;
        for (let i = 0; i < Math.min(leftJaw.length, rightJaw.length); i++) {
            symmetryScore += Math.abs(Math.abs(leftJaw[i].x - noseTip.x) - Math.abs(rightJaw[i].x - noseTip.x));
        }
        features.push(1 / (symmetryScore + 1));

        return features;
    }

    _getDistance(p1, p2) {
        const dx = p1.x - p2.x;
        const dy = p1.y - p2.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    _getAngle(p1, p2, p3) {
        const v1 = { x: p1.x - p2.x, y: p1.y - p2.y };
        const v2 = { x: p3.x - p2.x, y: p3.y - p2.y };
        const dot = v1.x * v2.x + v1.y * v2.y;
        const len1 = Math.sqrt(v1.x * v1.x + v1.y * v1.y);
        const len2 = Math.sqrt(v2.x * v2.x + v2.y * v2.y);
        return Math.acos(dot / (len1 * len2 + 1e-6));
    }

    _getCentroid(points) {
        const sum = points.reduce((acc, p) => ({
            x: acc.x + p.x,
            y: acc.y + p.y
        }), { x: 0, y: 0 });
        return { x: sum.x / points.length, y: sum.y / points.length };
    }

    /**
     * Enhanced comparison using ensemble metrics (Cosine, Euclidean, Manhattan)
     */
    compareFaces(embedding1, embedding2) {
        if (!embedding1 || !embedding2) return 0;

        const length = Math.min(embedding1.length, embedding2.length);
        let cosineSim = 0;
        let diffSqrSum = 0;
        let absDiffSum = 0;
        let maxDeviation = 0;

        for (let i = 0; i < length; i++) {
            cosineSim += embedding1[i] * embedding2[i];
            const diff = embedding1[i] - embedding2[i];
            diffSqrSum += diff * diff;
            absDiffSum += Math.abs(diff);

            const relativeDiff = Math.abs(diff) / (Math.abs(embedding1[i]) + 1e-6);
            if (relativeDiff > maxDeviation) maxDeviation = relativeDiff;
        }

        const euclideanSim = 1 / (1 + Math.sqrt(diffSqrSum));
        const manhattanSim = 1 / (1 + absDiffSum);

        let finalSimilarity = (cosineSim * 0.6 + euclideanSim * 0.25 + manhattanSim * 0.15);

        // Strictness Penalty: Prevents matches if individual features deviate too far
        if (maxDeviation > 0.30) {
            const hasSignificantAbsDiff = embedding1.some((val, idx) => Math.abs(val - embedding2[idx]) > 0.04);
            if (hasSignificantAbsDiff) {
                const penalty = Math.min((maxDeviation - 0.30) * 0.4, 0.12);
                finalSimilarity -= penalty;
            }
        }

        return Math.max(0, finalSimilarity);
    }

    /**
     * Calibrated verification for 270KB model
     */
    isSamePerson(embedding1, embedding2, threshold = 0.78) {
        const similarity = this.compareFaces(embedding1, embedding2);
        let level = 'low';
        let isSame = false;

        if (similarity >= 0.90) {
            level = 'very_high';
            isSame = true;
        } else if (similarity >= 0.84) {
            level = 'high';
            isSame = true;
        } else if (similarity >= threshold) {
            level = 'medium';
            isSame = true;
        } else if (similarity >= 0.60) {
            level = 'uncertain';
            isSame = false;
        } else {
            level = 'low';
            isSame = false;
        }

        return { isSame, confidence: similarity, level };
    }

    dispose() {
        this.isReady = false;
        this.modelsLoaded = false;
    }
}

export default FaceApiRecognitionModel;
