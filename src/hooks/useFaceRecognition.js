/**
 * useFaceRecognition Hook
 * React hook for face recognition functionality using face-api.js
 */

import { useState, useEffect, useRef, useCallback } from 'react';
import FaceApiRecognitionModel from '../models/FaceRecognitionModel';
import { MODEL_CONFIG } from '../config/modelConfig';

export function useFaceRecognition() {
    const [isLoading, setIsLoading] = useState(true);
    const [isModelReady, setIsModelReady] = useState(false);
    const [error, setError] = useState(null);
    const [verificationResult, setVerificationResult] = useState(null);
    const [isVerifying, setIsVerifying] = useState(false);

    const [unauthorizedLogs, setUnauthorizedLogs] = useState(() => {
        const saved = localStorage.getItem('face_sentinel_unauthorized_logs');
        return saved ? JSON.parse(saved) : [];
    });

    const modelRef = useRef(null);
    const referenceEmbeddingRef = useRef(null);
    const unauthorizedEmbeddingsRef = useRef([]); // Track unique unauthorized persons
    const verificationIntervalRef = useRef(null);
    const videoRef = useRef(null);

    // Persist logs
    useEffect(() => {
        localStorage.setItem('face_sentinel_unauthorized_logs', JSON.stringify(unauthorizedLogs));
    }, [unauthorizedLogs]);

    // Initialize model on mount
    useEffect(() => {
        const initModel = async () => {
            try {
                setIsLoading(true);
                setError(null);

                modelRef.current = new FaceApiRecognitionModel();
                await modelRef.current.loadModel();

                setIsModelReady(true);
                setIsLoading(false);
            } catch (err) {
                console.error('Failed to initialize model:', err);
                setError('Failed to load face recognition model: ' + err.message);
                setIsLoading(false);
            }
        };

        initModel();

        // Cleanup on unmount
        return () => {
            if (modelRef.current) {
                modelRef.current.dispose();
            }
            if (verificationIntervalRef.current) {
                clearInterval(verificationIntervalRef.current);
            }
        };
    }, []);

    /**
     * Add an unauthorized encounter to the logs
     */
    const logUnauthorizedEncounter = useCallback((embedding) => {
        let personId = -1;

        // Check if this is a person we've seen before (unauthorized)
        for (let i = 0; i < unauthorizedEmbeddingsRef.current.length; i++) {
            const comp = modelRef.current.isSamePerson(unauthorizedEmbeddingsRef.current[i], embedding);
            if (comp.isSame) {
                personId = i + 1;
                break;
            }
        }

        // If new person, add to reference list
        if (personId === -1) {
            unauthorizedEmbeddingsRef.current.push(embedding);
            personId = unauthorizedEmbeddingsRef.current.length;
        }

        const newLog = {
            id: Date.now(),
            timestamp: new Date().toLocaleTimeString(),
            fullDate: new Date().toLocaleString(),
            personLabel: `Unauthorized Person #${personId}`,
            personId: personId
        };

        setUnauthorizedLogs(prev => [newLog, ...prev].slice(0, 100)); // Keep last 100
    }, []);

    /**
     * Enroll a face as the reference for verification
     * @param {HTMLVideoElement} videoElement - Video element to capture from
     */
    const enrollFace = useCallback(async (videoElement) => {
        if (!isModelReady) {
            throw new Error('Model not ready');
        }

        try {
            setError(null);
            console.log('Enrolling face...');

            const result = await modelRef.current.extractEmbedding(videoElement);

            if (!result || !result.embedding) {
                throw new Error('No face detected. Please ensure your face is visible in the camera.');
            }

            // Check liveness
            if (!result.liveness.isLive) {
                throw new Error(`⚠️ Photo detected! ${result.liveness.reason}\n\nPlease use a real face, not a photo or screen.`);
            }

            const embedding = result.embedding;

            // Store new reference embedding (arrays don't need disposal)
            referenceEmbeddingRef.current = embedding;
            console.log('Face enrolled successfully');

            return true;
        } catch (err) {
            console.error('Face enrollment failed:', err);
            setError(err.message);
            throw err;
        }
    }, [isModelReady]);

    /**
     * Verify current face against reference
     * @param {HTMLVideoElement} videoElement - Video element to verify
     */
    const verifySingleFrame = useCallback(async (videoElement) => {
        if (!isModelReady || !referenceEmbeddingRef.current) {
            return null;
        }

        try {
            const result = await modelRef.current.extractEmbedding(videoElement);

            if (!result || !result.embedding) {
                // Clear verification result when no face is detected
                setVerificationResult(null);
                console.warn('No face detected in frame');
                return null;
            }

            const currentEmbedding = result.embedding;
            const livenessResult = result.liveness;

            // Check liveness first
            if (!livenessResult.isLive) {
                setVerificationResult({
                    isSame: false,
                    confidence: 0,
                    level: 'spoofing',
                    message: `⚠️ Photo/Screen Detected!`,
                    details: livenessResult.reason
                });
                return { isSame: false, spoofing: true };
            }

            const comparisonResult = modelRef.current.isSamePerson(
                referenceEmbeddingRef.current,
                currentEmbedding
            );

            const verifyResult = {
                ...comparisonResult,
                message: comparisonResult.isSame
                    ? `Same Person (${(comparisonResult.confidence * 100).toFixed(1)}%)`
                    : `Different Person (${(comparisonResult.confidence * 100).toFixed(1)}%)`
            };

            setVerificationResult(verifyResult);

            // Log unauthorized access
            if (!comparisonResult.isSame) {
                logUnauthorizedEncounter(currentEmbedding);
            }

            return verifyResult;
        } catch (err) {
            console.error('Verification failed:', err);
            setError(err.message);
            return null;
        }
    }, [isModelReady, logUnauthorizedEncounter]);

    /**
     * Start continuous real-time verification
     * @param {HTMLVideoElement} videoElement - Video element to monitor
     */
    const startVerification = useCallback((videoElement) => {
        if (!isModelReady || !referenceEmbeddingRef.current) {
            setError('Please enroll a face first');
            return;
        }

        if (isVerifying) {
            console.log('Verification already running');
            return;
        }

        setIsVerifying(true);
        setError(null);
        videoRef.current = videoElement;

        console.log('Starting real-time verification...');

        // Verify immediately
        verifySingleFrame(videoElement);

        // Set up interval for continuous verification
        verificationIntervalRef.current = setInterval(() => {
            if (videoRef.current) {
                verifySingleFrame(videoRef.current);
            }
        }, MODEL_CONFIG.VERIFICATION_INTERVAL_MS);
    }, [isModelReady, isVerifying, verifySingleFrame]);

    /**
     * Stop continuous verification
     */
    const stopVerification = useCallback(() => {
        if (verificationIntervalRef.current) {
            clearInterval(verificationIntervalRef.current);
            verificationIntervalRef.current = null;
        }

        setIsVerifying(false);
        setVerificationResult(null);
        videoRef.current = null;

        console.log('Stopped verification');
    }, []);

    /**
     * Reset all state (clear reference embedding)
     */
    const reset = useCallback(() => {
        stopVerification();

        // Clear reference embedding (arrays don't need disposal)
        referenceEmbeddingRef.current = null;

        setVerificationResult(null);
        setError(null);
    }, [stopVerification]);

    /**
     * Clear all logs
     */
    const clearLogs = useCallback(() => {
        setUnauthorizedLogs([]);
        unauthorizedEmbeddingsRef.current = [];
        localStorage.removeItem('face_sentinel_unauthorized_logs');
    }, []);

    return {
        isLoading,
        isModelReady,
        error,
        verificationResult,
        isVerifying,
        hasReference: !!referenceEmbeddingRef.current,
        unauthorizedLogs,
        enrollFace,
        startVerification,
        stopVerification,
        verifySingleFrame,
        clearLogs,
        reset
    };
}

export default useFaceRecognition;
