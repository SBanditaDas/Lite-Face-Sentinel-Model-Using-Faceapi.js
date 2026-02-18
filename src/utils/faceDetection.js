/**
 * Face Detection Utilities using BlazeFace
 * Responsible for bounding box detection and ROI extraction.
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
        faceDetectionModel = await blazeface.load(MODEL_CONFIG.FACE_DETECTION_CONFIG);
    }
    return faceDetectionModel;
}

/**
 * Detect faces in an image or video frame
 */
export async function detectFaces(input) {
    if (!faceDetectionModel) await loadFaceDetector();
    return await faceDetectionModel.estimateFaces(input, false);
}

/**
 * Extract faceregion from image based on bounding box
 */
export function extractFaceROI(input, face) {
    const [x, y, width, height] = face.topLeft.concat(face.bottomRight).reduce((acc, val, i) => {
        if (i < 2) return [...acc, val];
        return [...acc, val - acc[i - 2]];
    }, []);

    const padding = 0.1;
    const paddedX = Math.max(0, x - width * padding);
    const paddedY = Math.max(0, y - height * padding);
    const paddedWidth = width * (1 + 2 * padding);
    const paddedHeight = height * (1 + 2 * padding);

    let imageTensor = tf.browser.fromPixels(input);

    const faceTensor = tf.image.cropAndResize(
        imageTensor.expandDims(0),
        [[paddedY / imageTensor.shape[0],
        paddedX / imageTensor.shape[1],
        (paddedY + paddedHeight) / imageTensor.shape[0],
        (paddedX + paddedWidth) / imageTensor.shape[1]]],
        [0],
        [MODEL_CONFIG.INPUT_SIZE, MODEL_CONFIG.INPUT_SIZE]
    );

    imageTensor.dispose();
    return faceTensor.squeeze([0]);
}

/**
 * Get the largest detected face (closest to camera)
 */
export function getLargestFace(faces) {
    if (!faces || faces.length === 0) return null;

    return faces.reduce((largest, face) => {
        const area = (face.bottomRight[0] - face.topLeft[0]) * (face.bottomRight[1] - face.topLeft[1]);
        const largestArea = (largest.bottomRight[0] - largest.topLeft[0]) * (largest.bottomRight[1] - largest.topLeft[1]);
        return area > largestArea ? face : largest;
    });
}

export function disposeDetector() {
    if (faceDetectionModel) {
        faceDetectionModel.dispose();
        faceDetectionModel = null;
    }
}
