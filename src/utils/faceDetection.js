/**
 * Face Detection Utilities using BlazeFace
 * Detects face bounding boxes in images/video frames
 */

import * as blazeface from '@tensorflow-models/blazeface';
import * as tf from '@tensorflow/tfjs';
import { MODEL_CONFIG } from '../config/modelConfig';

let faceDetectionModel = null;

/**
 * Load BlazeFace model for face detection
 */
export async function loadFaceDetector() {
    if (!faceDetectionModel) {
        console.log('Loading BlazeFace model...');
        faceDetectionModel = await blazeface.load(MODEL_CONFIG.FACE_DETECTION_CONFIG);
        console.log('BlazeFace model loaded successfully');
    }
    return faceDetectionModel;
}

/**
 * Detect faces in an image or video frame
 * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} input - Input element
 * @returns {Promise<Array>} Array of detected faces with bounding boxes
 */
export async function detectFaces(input) {
    if (!faceDetectionModel) {
        await loadFaceDetector();
    }

    const predictions = await faceDetectionModel.estimateFaces(input, false);
    return predictions;
}

/**
 * Extract face region from image based on bounding box
 * @param {HTMLVideoElement|HTMLImageElement|HTMLCanvasElement} input - Input element
 * @param {Object} face - Face detection result with bounding box
 * @returns {tf.Tensor3D} Cropped and resized face tensor (112x112x3)
 */
export function extractFaceROI(input, face) {
    const [x, y, width, height] = face.topLeft.concat(face.bottomRight).reduce((acc, val, i) => {
        if (i < 2) return [...acc, val];
        return [...acc, val - acc[i - 2]];
    }, []);

    // Add padding (10%) around face
    const padding = 0.1;
    const paddedX = Math.max(0, x - width * padding);
    const paddedY = Math.max(0, y - height * padding);
    const paddedWidth = width * (1 + 2 * padding);
    const paddedHeight = height * (1 + 2 * padding);

    // Convert input to tensor
    let imageTensor = tf.browser.fromPixels(input);

    // Crop face region
    const faceTensor = tf.image.cropAndResize(
        imageTensor.expandDims(0),
        [[paddedY / imageTensor.shape[0],
        paddedX / imageTensor.shape[1],
        (paddedY + paddedHeight) / imageTensor.shape[0],
        (paddedX + paddedWidth) / imageTensor.shape[1]]],
        [0],
        [MODEL_CONFIG.INPUT_SIZE, MODEL_CONFIG.INPUT_SIZE]
    );

    // Clean up
    imageTensor.dispose();

    return faceTensor.squeeze([0]);
}

/**
 * Get the largest detected face (closest to camera)
 * @param {Array} faces - Array of detected faces
 * @returns {Object|null} Largest face or null
 */
export function getLargestFace(faces) {
    if (!faces || faces.length === 0) return null;

    return faces.reduce((largest, face) => {
        const area = (face.bottomRight[0] - face.topLeft[0]) *
            (face.bottomRight[1] - face.topLeft[1]);
        const largestArea = (largest.bottomRight[0] - largest.topLeft[0]) *
            (largest.bottomRight[1] - largest.topLeft[1]);
        return area > largestArea ? face : largest;
    });
}

/**
 * Cleanup resources
 */
export function disposeDetector() {
    if (faceDetectionModel) {
        faceDetectionModel.dispose();
        faceDetectionModel = null;
    }
}
