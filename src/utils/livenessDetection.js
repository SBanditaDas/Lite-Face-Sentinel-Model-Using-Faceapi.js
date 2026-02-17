/**
 * Liveness Detection Utility
 * Detects photo spoofing attempts (phone screens, printed photos)
 * Uses texture analysis, color profiling, and glare detection
 */

/**
 * Extract image data from a region of interest
 */
function getImageData(videoElement, box) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');

    // Set canvas size to face bounding box
    canvas.width = box.width;
    canvas.height = box.height;

    // Draw the face region
    ctx.drawImage(
        videoElement,
        box.x, box.y, box.width, box.height,
        0, 0, box.width, box.height
    );

    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Analyze texture variance (real faces have more texture than photos)
 * Uses Laplacian variance to detect blur/flatness
 */
function analyzeTexture(videoElement, box) {
    try {
        const imageData = getImageData(videoElement, box);
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;

        // Calculate grayscale Laplacian variance
        let sum = 0;
        let count = 0;

        // Sample every 4th pixel for performance
        for (let y = 2; y < height - 2; y += 4) {
            for (let x = 2; x < width - 2; x += 4) {
                const idx = (y * width + x) * 4;

                // Convert to grayscale
                const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;

                // Laplacian kernel (edge detection)
                const neighbors = [
                    data[((y - 1) * width + x) * 4],     // top
                    data[((y + 1) * width + x) * 4],     // bottom
                    data[(y * width + (x - 1)) * 4],     // left
                    data[(y * width + (x + 1)) * 4]      // right
                ];

                const avgNeighbor = neighbors.reduce((a, b) => a + b, 0) / 4;
                const laplacian = Math.abs(gray - avgNeighbor);

                sum += laplacian * laplacian;
                count++;
            }
        }

        const variance = count > 0 ? sum / count : 0;

        // Normalize to 0-1 (higher = more texture = more likely real)
        // Real faces: variance typically 20-100
        // Photos/screens: variance typically 5-20
        const score = Math.min(variance / 30, 1.0);

        console.log('Texture score:', score.toFixed(3), 'variance:', variance.toFixed(2));
        return score;

    } catch (error) {
        console.error('Texture analysis failed:', error);
        return 0.5; // Neutral score on error
    }
}

/**
 * Analyze color profile (screens have different RGB distribution)
 */
function analyzeColorProfile(videoElement, box) {
    try {
        const imageData = getImageData(videoElement, box);
        const data = imageData.data;

        let rSum = 0, gSum = 0, bSum = 0;
        let rVar = 0, gVar = 0, bVar = 0;
        let count = 0;

        // Calculate RGB means
        for (let i = 0; i < data.length; i += 16) { // Sample every 4th pixel
            rSum += data[i];
            gSum += data[i + 1];
            bSum += data[i + 2];
            count++;
        }

        const rMean = rSum / count;
        const gMean = gSum / count;
        const bMean = bSum / count;

        // Calculate RGB variances
        for (let i = 0; i < data.length; i += 16) {
            rVar += Math.pow(data[i] - rMean, 2);
            gVar += Math.pow(data[i + 1] - gMean, 2);
            bVar += Math.pow(data[i + 2] - bMean, 2);
        }

        rVar /= count;
        gVar /= count;
        bVar /= count;

        // Real faces have balanced RGB variance (skin tones)
        // Screens often have blue shift or unnatural color uniformity
        const avgVar = (rVar + gVar + bVar) / 3;
        const varBalance = 1 - (Math.abs(rVar - gVar) + Math.abs(gVar - bVar) + Math.abs(bVar - rVar)) / (3 * avgVar + 1);

        // Check for blue shift (common in screens)
        const blueShift = bMean > (rMean + gMean) / 2 ? 0.7 : 1.0;

        const score = varBalance * blueShift * 0.8 + 0.2; // Weighted

        console.log('Color score:', score.toFixed(3), 'balance:', varBalance.toFixed(3));
        return Math.max(0, Math.min(1, score));

    } catch (error) {
        console.error('Color analysis failed:', error);
        return 0.5;
    }
}

/**
 * Detect screen glare/reflections (rectangular highlights on phone screens)
 */
function detectScreenGlare(videoElement, box) {
    try {
        const imageData = getImageData(videoElement, box);
        const data = imageData.data;
        const width = imageData.width;
        const height = imageData.height;

        // Count bright pixels (potential glare)
        let brightPixels = 0;
        let totalBrightness = 0;

        for (let i = 0; i < data.length; i += 16) {
            const brightness = (data[i] + data[i + 1] + data[i + 2]) / 3;
            if (brightness > 220) { // Very bright pixel
                brightPixels++;
                totalBrightness += brightness;
            }
        }

        const brightRatio = brightPixels / (data.length / 16);

        // Screens often have uniform bright reflections (5-15% of pixels)
        // Real faces rarely have this pattern
        const score = brightRatio > 0.05 && brightRatio < 0.20 ? 0.6 : 1.0;

        console.log('Glare score:', score.toFixed(3), 'bright ratio:', brightRatio.toFixed(4));
        return score;

    } catch (error) {
        console.error('Glare detection failed:', error);
        return 0.5;
    }
}

/**
 * Track landmark history for motion detection
 */
let landmarkHistory = [];
const MAX_HISTORY = 5;

/**
 * Check for micro-movements (real faces slightly move, photos are static)
 */
function checkMicroMotion(landmarks) {
    try {
        // Get key landmark positions (eyes, nose)
        const positions = landmarks.positions;
        const keyPoints = [
            positions[27], // nose tip
            positions[36], // left eye outer corner
            positions[45]  // right eye outer corner
        ];

        // Store current positions
        landmarkHistory.push(keyPoints);
        if (landmarkHistory.length > MAX_HISTORY) {
            landmarkHistory.shift();
        }

        // Need at least 3 frames to detect motion
        if (landmarkHistory.length < 3) {
            return 0.5; // Neutral until we have enough data
        }

        // Calculate average movement
        let totalMovement = 0;
        for (let i = 1; i < landmarkHistory.length; i++) {
            for (let j = 0; j < keyPoints.length; j++) {
                const dx = landmarkHistory[i][j].x - landmarkHistory[i - 1][j].x;
                const dy = landmarkHistory[i][j].y - landmarkHistory[i - 1][j].y;
                totalMovement += Math.sqrt(dx * dx + dy * dy);
            }
        }

        const avgMovement = totalMovement / ((landmarkHistory.length - 1) * keyPoints.length);

        // Real faces: 0.5-3 pixels movement per frame
        // Static photos: <0.3 pixels (just detection noise)
        // Videos on screens: Often >3 pixels (shakier)
        const score = avgMovement > 0.3 && avgMovement < 5 ? 1.0 : 0.6;

        console.log('Motion score:', score.toFixed(3), 'avg movement:', avgMovement.toFixed(3));
        return score;

    } catch (error) {
        console.error('Motion detection failed:', error);
        return 0.5;
    }
}

/**
 * Main liveness detection function
 * Returns score and whether face appears to be live (not a photo)
 */
export async function detectLiveness(videoElement, landmarks, detection) {
    try {
        const box = detection.box;

        // Run all detection methods
        const textureScore = analyzeTexture(videoElement, box);
        const colorScore = analyzeColorProfile(videoElement, box);
        const glareScore = detectScreenGlare(videoElement, box);
        const motionScore = checkMicroMotion(landmarks);

        // Weighted ensemble
        const livenessScore = (
            textureScore * 0.40 +  // Texture is most important
            colorScore * 0.30 +    // Color profile second
            glareScore * 0.20 +    // Glare detection
            motionScore * 0.10     // Motion helps but less reliable
        );

        const isLive = livenessScore >= 0.60; // Threshold

        console.log('=== LIVENESS DETECTION ===');
        console.log('  Texture:', (textureScore * 100).toFixed(1) + '%');
        console.log('  Color:', (colorScore * 100).toFixed(1) + '%');
        console.log('  Glare:', (glareScore * 100).toFixed(1) + '%');
        console.log('  Motion:', (motionScore * 100).toFixed(1) + '%');
        console.log('  FINAL:', (livenessScore * 100).toFixed(1) + '%', isLive ? '✓ LIVE' : '⚠️ PHOTO');
        console.log('==========================');

        return {
            isLive: isLive,
            confidence: livenessScore,
            scores: {
                texture: textureScore,
                color: colorScore,
                glare: glareScore,
                motion: motionScore
            },
            reason: isLive
                ? 'Live face detected'
                : 'Possible photo/screen detected (failed liveness check)'
        };

    } catch (error) {
        console.error('Liveness detection error:', error);
        // On error, assume live (fail open for better UX)
        return {
            isLive: true,
            confidence: 0.5,
            scores: {},
            reason: 'Liveness check error - defaulting to live'
        };
    }
}

/**
 * Reset motion history (call when starting new verification session)
 */
export function resetLivenessDetection() {
    landmarkHistory = [];
    console.log('Liveness detection reset');
}
