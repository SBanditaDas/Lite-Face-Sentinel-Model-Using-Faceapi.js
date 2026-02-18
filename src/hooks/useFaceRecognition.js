/**
 * useFaceRecognition Hook
 * Manages model lifecycle, face enrollment, and real-time verification results.
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
    const unauthorizedEmbeddingsRef = useRef([]);
    const verificationIntervalRef = useRef(null);
    const videoRef = useRef(null);

    useEffect(() => {
        localStorage.setItem('face_sentinel_unauthorized_logs', JSON.stringify(unauthorizedLogs));
    }, [unauthorizedLogs]);

    useEffect(() => {
        const initModel = async () => {
            try {
                setIsLoading(true);
                modelRef.current = new FaceApiRecognitionModel();
                await modelRef.current.loadModel();
                setIsModelReady(true);
                setIsLoading(false);
            } catch (err) {
                setError('Failed to load face recognition model: ' + err.message);
                setIsLoading(false);
            }
        };

        initModel();

        return () => {
            if (modelRef.current) modelRef.current.dispose();
            if (verificationIntervalRef.current) clearInterval(verificationIntervalRef.current);
        };
    }, []);

    /**
     * Clusters unauthorized faces to identify unique individuals
     */
    const logUnauthorizedEncounter = useCallback((embedding) => {
        let personId = -1;

        for (let i = 0; i < unauthorizedEmbeddingsRef.current.length; i++) {
            const comp = modelRef.current.isSamePerson(unauthorizedEmbeddingsRef.current[i], embedding);
            if (comp.isSame) {
                personId = i + 1;
                break;
            }
        }

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

        setUnauthorizedLogs(prev => [newLog, ...prev].slice(0, 100));
    }, []);

    /**
     * Capture reference face for future verification
     */
    const enrollFace = useCallback(async (videoElement) => {
        if (!isModelReady) throw new Error('Model not ready');

        try {
            setError(null);
            const result = await modelRef.current.extractEmbedding(videoElement);

            if (!result || !result.embedding) {
                throw new Error('No face detected. Please ensure your face is visible in the camera.');
            }

            if (!result.liveness.isLive) {
                throw new Error(`⚠️ Photo detected! ${result.liveness.reason}`);
            }

            referenceEmbeddingRef.current = result.embedding;
            return true;
        } catch (err) {
            setError(err.message);
            throw err;
        }
    }, [isModelReady]);

    /**
     * Checks a single frame against the reference embedding
     */
    const verifySingleFrame = useCallback(async (videoElement) => {
        if (!isModelReady || !referenceEmbeddingRef.current) return null;

        try {
            const result = await modelRef.current.extractEmbedding(videoElement);

            if (!result || !result.embedding) {
                setVerificationResult(null);
                return null;
            }

            const currentEmbedding = result.embedding;
            const livenessResult = result.liveness;

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

            if (!comparisonResult.isSame) {
                logUnauthorizedEncounter(currentEmbedding);
            }

            return verifyResult;
        } catch (err) {
            setError(err.message);
            return null;
        }
    }, [isModelReady, logUnauthorizedEncounter]);

    /**
     * Starts continuous background verification
     */
    const startVerification = useCallback((videoElement) => {
        if (!isModelReady || !referenceEmbeddingRef.current) {
            setError('Please enroll a face first');
            return;
        }

        if (isVerifying) return;

        setIsVerifying(true);
        setError(null);
        videoRef.current = videoElement;

        verifySingleFrame(videoElement);

        verificationIntervalRef.current = setInterval(() => {
            if (videoRef.current) verifySingleFrame(videoRef.current);
        }, MODEL_CONFIG.VERIFICATION_INTERVAL_MS);
    }, [isModelReady, isVerifying, verifySingleFrame]);

    /**
     * Stops the continuous verification loop
     */
    const stopVerification = useCallback(() => {
        if (verificationIntervalRef.current) {
            clearInterval(verificationIntervalRef.current);
            verificationIntervalRef.current = null;
        }

        setIsVerifying(false);
        setVerificationResult(null);
        videoRef.current = null;
    }, []);

    /**
     * Clears reference data and resets state
     */
    const reset = useCallback(() => {
        stopVerification();
        referenceEmbeddingRef.current = null;
        setVerificationResult(null);
        setError(null);
    }, [stopVerification]);

    /**
     * Clears all security logs from memory and storage
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
