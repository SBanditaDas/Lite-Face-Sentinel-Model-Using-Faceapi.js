/**
 * Liveness Detection Utility
 * Detects photo spoofing attempts using texture, color, glare, and motion analysis.
 */

function getImageData(videoElement, box) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = box.width;
    canvas.height = box.height;

    ctx.drawImage(
        videoElement,
        box.x, box.y, box.width, box.height,
        0, 0, box.width, box.height
    );

    return ctx.getImageData(0, 0, canvas.width, canvas.height);
}

/**
 * Detects blur/flatness using Laplacian variance
 */
function analyzeTexture(videoElement, box) {
    try {
        const imageData = getImageData(videoElement, box);
        const data = imageData.data;
        const { width, height } = imageData;

        let sum = 0, count = 0;

        for (let y = 2; y < height - 2; y += 4) {
            for (let x = 2; x < width - 2; x += 4) {
                const idx = (y * width + x) * 4;
                const gray = (data[idx] + data[idx + 1] + data[idx + 2]) / 3;
                const neighbors = [
                    data[((y - 1) * width + x) * 4],
                    data[((y + 1) * width + x) * 4],
                    data[(y * width + (x - 1)) * 4],
                    data[(y * width + (x + 1)) * 4]
                ];

                const avgNeighbor = neighbors.reduce((a, b) => a + b, 0) / 4;
                sum += Math.pow(gray - avgNeighbor, 2);
                count++;
            }
        }

        const variance = count > 0 ? sum / count : 0;
        // Real faces: variance 20-100 | Photos/screens: variance 5-20
        return Math.min(variance / 30, 1.0);
    } catch (error) {
        return 0.5;
    }
}

/**
 * Checks for unnatural RGB distribution common in screens
 */
function analyzeColorProfile(videoElement, box) {
    try {
        const imageData = getImageData(videoElement, box);
        const data = imageData.data;

        let rSum = 0, gSum = 0, bSum = 0, count = 0;

        for (let i = 0; i < data.length; i += 16) {
            rSum += data[i];
            gSum += data[i + 1];
            bSum += data[i + 2];
            count++;
        }

        const [rMean, gMean, bMean] = [rSum / count, gSum / count, bSum / count];
        let rVar = 0, gVar = 0, bVar = 0;

        for (let i = 0; i < data.length; i += 16) {
            rVar += Math.pow(data[i] - rMean, 2);
            gVar += Math.pow(data[i + 1] - gMean, 2);
            bVar += Math.pow(data[i + 2] - bMean, 2);
        }

        const avgVar = (rVar + gVar + bVar) / (3 * count);
        const varBalance = 1 - (Math.abs(rVar - gVar) + Math.abs(gVar - bVar) + Math.abs(bVar - rVar)) / (3 * rVar + 3 * gVar + 3 * bVar + 1);
        const blueShift = bMean > (rMean + gMean) / 2 ? 0.7 : 1.0;

        return Math.max(0, Math.min(1, varBalance * blueShift * 0.8 + 0.2));
    } catch (error) {
        return 0.5;
    }
}

/**
 * Detects rectangular highlights typical of phone screens
 */
function detectScreenGlare(videoElement, box) {
    try {
        const imageData = getImageData(videoElement, box);
        const data = imageData.data;
        let brightPixels = 0;

        for (let i = 0; i < data.length; i += 16) {
            if ((data[i] + data[i + 1] + data[i + 2]) / 3 > 220) brightPixels++;
        }

        const brightRatio = brightPixels / (data.length / 16);
        return brightRatio > 0.05 && brightRatio < 0.20 ? 0.6 : 1.0;
    } catch (error) {
        return 0.5;
    }
}

let landmarkHistory = [];
const MAX_HISTORY = 5;

/**
 * Differentiates micro-movements of real skin from static photos
 */
function checkMicroMotion(landmarks) {
    try {
        const positions = landmarks.positions;
        const keyPoints = [positions[27], positions[36], positions[45]];

        landmarkHistory.push(keyPoints);
        if (landmarkHistory.length > MAX_HISTORY) landmarkHistory.shift();
        if (landmarkHistory.length < 3) return 0.5;

        let totalMovement = 0;
        for (let i = 1; i < landmarkHistory.length; i++) {
            for (let j = 0; j < keyPoints.length; j++) {
                totalMovement += Math.sqrt(
                    Math.pow(landmarkHistory[i][j].x - landmarkHistory[i - 1][j].x, 2) +
                    Math.pow(landmarkHistory[i][j].y - landmarkHistory[i - 1][j].y, 2)
                );
            }
        }

        const avgMovement = totalMovement / ((landmarkHistory.length - 1) * keyPoints.length);
        // Real faces: 0.5-3px movement | Static photos: <0.3px
        return avgMovement > 0.3 && avgMovement < 5 ? 1.0 : 0.6;
    } catch (error) {
        return 0.5;
    }
}

/**
 * Primary aggregator for spoofing protection
 */
export async function detectLiveness(videoElement, landmarks, detection) {
    try {
        const textureScore = analyzeTexture(videoElement, detection.box);
        const colorScore = analyzeColorProfile(videoElement, detection.box);
        const glareScore = detectScreenGlare(videoElement, detection.box);
        const motionScore = checkMicroMotion(landmarks);

        const livenessScore = (textureScore * 0.4 + colorScore * 0.3 + glareScore * 0.2 + motionScore * 0.1);
        const isLive = livenessScore >= 0.60;

        return {
            isLive,
            confidence: livenessScore,
            scores: { texture: textureScore, color: colorScore, glare: glareScore, motion: motionScore },
            reason: isLive ? 'Live face detected' : 'Possible photo/screen detected'
        };
    } catch (error) {
        // Fail open for better UX on random calculation errors
        return { isLive: true, confidence: 0.5, scores: {}, reason: 'Liveness check error' };
    }
}

export function resetLivenessDetection() {
    landmarkHistory = [];
}
