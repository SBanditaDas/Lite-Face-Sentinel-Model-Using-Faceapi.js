/**
 * FaceVerification Component
 * Main interface for face enrollment, real-time verification, and security logging.
 */

import React, { useRef, useState } from 'react';
import Webcam from 'react-webcam';
import { useFaceRecognition } from '../hooks/useFaceRecognition';
import './FaceVerification.css';

function FaceVerification() {
    const webcamRef = useRef(null);
    const [webcamReady, setWebcamReady] = useState(false);
    const [enrolled, setEnrolled] = useState(false);

    const {
        isLoading,
        isModelReady,
        error,
        verificationResult,
        isVerifying,
        hasReference,
        unauthorizedLogs,
        enrollFace,
        startVerification,
        stopVerification,
        clearLogs,
        reset
    } = useFaceRecognition();

    const handleWebcamReady = () => setWebcamReady(true);

    const handleEnroll = async () => {
        if (webcamRef.current && webcamRef.current.video) {
            try {
                await enrollFace(webcamRef.current.video);
                setEnrolled(true);
                alert('✓ Face enrolled successfully! You can now start verification.');
            } catch (err) {
                alert(`❌ Enrollment failed: ${err.message}`);
            }
        } else {
            alert('❌ Webcam not ready. Please wait a moment and try again.');
        }
    };

    const handleStartVerification = () => {
        if (webcamRef.current && webcamRef.current.video) {
            startVerification(webcamRef.current.video);
        }
    };

    const handleReset = () => {
        setEnrolled(false);
        reset();
    };

    const getBorderClass = () => {
        if (!verificationResult) return '';
        return verificationResult.isSame ? 'border-success' : 'border-danger';
    };

    const getConfidenceClass = () => {
        if (!verificationResult) return '';
        return `confidence-${verificationResult.level}`;
    };

    return (
        <div className="face-verification-container">
            <div className="face-verification-card">
                <h1>Face Sentinel</h1>
                <p className="subtitle">Lightweight Face Recognition</p>

                {/* Loading State */}
                {isLoading && (
                    <div className="status-message info">
                        <div className="spinner"></div>
                        <p>Loading face recognition model...</p>
                    </div>
                )}

                {/* Error State */}
                {error && (
                    <div className="status-message error">
                        <p>❌ {error}</p>
                    </div>
                )}

                {/* Webcam View */}
                {isModelReady && (
                    <div className={`webcam-container ${getBorderClass()}`}>
                        <Webcam
                            ref={webcamRef}
                            audio={false}
                            screenshotFormat="image/jpeg"
                            videoConstraints={{
                                width: 640,
                                height: 480,
                                facingMode: 'user'
                            }}
                            onUserMedia={handleWebcamReady}
                            className="webcam-feed"
                        />

                        {/* Verification Status Overlay */}
                        {verificationResult && (
                            <div className={`verification-overlay ${getConfidenceClass()}`}>
                                <div className="verification-status">
                                    {verificationResult.isSame ? '✓' : '✗'}
                                </div>
                                <div className="verification-message">
                                    {verificationResult.message}
                                </div>
                            </div>
                        )}

                        {isVerifying && !verificationResult && (
                            <div className="verification-overlay">
                                <div className="verification-message" style={{ color: '#ffa500' }}>
                                    ⚠️ No face detected
                                </div>
                            </div>
                        )}

                        {!webcamReady && (
                            <div className="webcam-placeholder">
                                <p>Initializing camera...</p>
                            </div>
                        )}
                    </div>
                )}

                {/* Controls */}
                {isModelReady && webcamReady && (
                    <div className="controls">
                        {!enrolled && !hasReference && (
                            <button className="btn btn-primary" onClick={handleEnroll}>
                                📸 Enroll Face
                            </button>
                        )}

                        {(enrolled || hasReference) && !isVerifying && (
                            <>
                                <button className="btn btn-success" onClick={handleStartVerification}>
                                    ▶ Start Verification
                                </button>
                                <button className="btn btn-secondary" onClick={handleReset}>
                                    🔄 Reset
                                </button>
                            </>
                        )}

                        {isVerifying && (
                            <button className="btn btn-warning" onClick={stopVerification}>
                                ⏸ Stop Verification
                            </button>
                        )}
                    </div>
                )}

                {/* Security Dashboard */}
                {enrolled && (
                    <div className="security-dashboard">
                        <div className="dashboard-header">
                            <h3>🛡️ Security Logs</h3>
                            <div className="dashboard-stats">
                                <div className="stat-item">
                                    <span className="stat-value">{unauthorizedLogs.length}</span>
                                    <span className="stat-label">Encounters</span>
                                </div>
                                <div className="stat-item">
                                    <span className="stat-value">
                                        {new Set(unauthorizedLogs.map(l => l.personId)).size}
                                    </span>
                                    <span className="stat-label">Unique Persons</span>
                                </div>
                            </div>
                        </div>

                        <div className="logs-container">
                            {unauthorizedLogs.length > 0 ? (
                                <table className="logs-table">
                                    <thead>
                                        <tr>
                                            <th>Time</th>
                                            <th>Identity</th>
                                            <th>Status</th>
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {unauthorizedLogs.slice(0, 10).map(log => (
                                            <tr key={log.id}>
                                                <td>{log.timestamp}</td>
                                                <td>{log.personLabel}</td>
                                                <td className="status-danger">Unauthorized</td>
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            ) : (
                                <p className="no-logs">No unauthorized encounters recorded yet.</p>
                            )}
                        </div>

                        {unauthorizedLogs.length > 0 && (
                            <button className="btn btn-clear-logs" onClick={clearLogs}>
                                🗑️ Clear Logs
                            </button>
                        )}
                    </div>
                )}

                {/* Model Info */}
                <div className="model-info">
                    <p className="info-text">
                        <strong>Model:</strong> face-api.js (Enhanced Landmarks)
                    </p>
                    <p className="info-text">
                        <strong>Size:</strong> ~270 KB | <strong>Accuracy:</strong> 94-97%
                    </p>
                </div>
            </div>
        </div>
    );
}

export default FaceVerification;
