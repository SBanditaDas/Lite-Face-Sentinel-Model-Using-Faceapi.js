# Integration Guide - Face Sentinel

This guide explains how to integrate Face Sentinel into your existing React application.

## Quick Integration (< 5 minutes)

### Step 1: Install Dependencies

```bash
npm install @tensorflow/tfjs @tensorflow-models/blazeface react-webcam
```

### Step 2: Copy Files

Copy these files from Face Sentinel to your project:

```
your-app/
├── src/
│   ├── components/
│   │   ├── FaceVerification.jsx
│   │   └── FaceVerification.css
│   ├── hooks/
│   │   └── useFaceRecognition.js
│   ├── models/
│   │   └── FaceRecognitionModel.js
│   ├── utils/
│   │   └── faceDetection.js
│   └── config/
│       └── modelConfig.js
└── public/
    └── models/
        └── face_recognition/
            ├── model.json
            └── *.bin files
```

### Step 3: Use in Your App

```jsx
import FaceVerification from './components/FaceVerification';

function YourApp() {
  return (
    <div>
      <h1>Your Application</h1>
      <FaceVerification />
    </div>
  );
}
```

Done! The component is fully self-contained.

---

## Custom Integration

If you need custom UI or workflow, use the hook directly:

### Example: Custom UI Component

```jsx
import { useRef } from 'react';
import Webcam from 'react-webcam';
import { useFaceRecognition } from './hooks/useFaceRecognition';

function CustomFaceAuth() {
  const webcamRef = useRef(null);
  const {
    isModelReady,
    enrollFace,
    verificationResult,
    startVerification,
  } = useFaceRecognition();

  const handleEnroll = async () => {
    if (webcamRef.current?.video) {
      await enrollFace(webcamRef.current.video);
      alert('Face enrolled!');
    }
  };

  return (
    <div>
      <Webcam ref={webcamRef} />
      
      {isModelReady && (
        <button onClick={handleEnroll}>
          Enroll My Face
        </button>
      )}
      
      {verificationResult && (
        <p>{verificationResult.message}</p>
      )}
    </div>
  );
}
```

### Example: Authentication Flow

```jsx
import { useFaceRecognition } from './hooks/useFaceRecognition';

function FaceAuthLogin() {
  const {
    enrollFace,
    verifySingleFrame,
    verificationResult
  } = useFaceRecognition();

  // 1. During registration: enroll face
  const handleRegister = async (videoElement) => {
    await enrollFace(videoElement);
    // Save state that user has enrolled face
    localStorage.setItem('faceEnrolled', 'true');
  };

  // 2. During login: verify once
  const handleLogin = async (videoElement) => {
    const result = await verifySingleFrame(videoElement);
    
    if (result?.isSame && result.confidence > 0.7) {
      // Login successful
      console.log('Face verified! Logging in...');
      // Proceed with login
    } else {
      alert('Face verification failed');
    }
  };

  return (
    // Your login UI
  );
}
```

---

## Configuration Options

### Adjust Similarity Threshold

Edit `src/config/modelConfig.js`:

```javascript
export const MODEL_CONFIG = {
  SIMILARITY_THRESHOLD: 0.6, // Default
  // Lower (0.4-0.5) = More lenient, may accept similar faces
  // Higher (0.7-0.8) = More strict, may reject same person
};
```

**Recommended values:**
- **0.5-0.6**: General use, balanced
- **0.7-0.8**: High security, stricter matching
- **0.4-0.5**: Convenience, more lenient (use with caution)

### Adjust Verification Speed

```javascript
export const MODEL_CONFIG = {
  VERIFICATION_INTERVAL_MS: 500, // Check every 500ms
  // Lower = Faster updates, more CPU usage
  // Higher = Slower updates, less CPU usage
};
```

**Recommended values:**
- **300-500ms**: Real-time monitoring
- **1000-2000ms**: Periodic checks
- **100-200ms**: Very responsive (high CPU usage)

---

## API Reference

### useFaceRecognition Hook

```typescript
const {
  // State
  isLoading: boolean,          // Model is loading
  isModelReady: boolean,       // Model ready for use
  error: string | null,        // Error message
  isVerifying: boolean,        // Verification in progress
  hasReference: boolean,       // Reference face enrolled
  verificationResult: {        // Latest verification result
    isSame: boolean,           // Same person?
    confidence: number,        // 0-1 similarity score
    level: 'low'|'medium'|'high', // Confidence level
    message: string            // Human-readable message
  } | null,
  
  // Methods
  enrollFace: (videoElement) => Promise<boolean>,
  startVerification: (videoElement) => void,
  stopVerification: () => void,
  verifySingleFrame: (videoElement) => Promise<Result>,
  reset: () => void
} = useFaceRecognition();
```

### FaceRecognitionModel Class

```javascript
import FaceRecognitionModel from './models/FaceRecognitionModel';

const model = new FaceRecognitionModel();

// Load model
await model.loadModel();

// Extract embedding from image/video
const embedding = await model.extractEmbedding(imageElement);

// Compare two embeddings
const similarity = model.compareFaces(embedding1, embedding2);

// Check if same person
const result = model.isSamePerson(embedding1, embedding2, threshold);
// Returns: { isSame, confidence, level }
```

---

## Performance Optimization

### 1. Lazy Load Model

Load model only when needed:

```jsx
function App() {
  const [showFaceAuth, setShowFaceAuth] = useState(false);

  return (
    <div>
      <button onClick={() => setShowFaceAuth(true)}>
        Enable Face Auth
      </button>
      
      {showFaceAuth && <FaceVerification />}
    </div>
  );
}
```

### 2. Reduce Video Resolution

Lower resolution = faster processing:

```jsx
<Webcam
  videoConstraints={{
    width: 320,  // Lower than default 640
    height: 240, // Lower than default 480
    facingMode: 'user'
  }}
/>
```

### 3. Use Web Workers (Advanced)

Offload inference to worker thread:

```javascript
// worker.js
import * as tf from '@tensorflow/tfjs';
import FaceRecognitionModel from './models/FaceRecognitionModel';

// Handle messages from main thread
self.onmessage = async (e) => {
  const { type, data } = e.data;
  
  if (type === 'EXTRACT_EMBEDDING') {
    const result = await model.extractEmbedding(data);
    self.postMessage({ type: 'EMBEDDING_RESULT', data: result });
  }
};
```

---

## Security Considerations

### 1. HTTPS Required

Webcam access requires secure context:

```nginx
# Nginx config
server {
  listen 443 ssl;
  ssl_certificate /path/to/cert.pem;
  ssl_certificate_key /path/to/key.pem;
  
  location / {
    # Your app
  }
}
```

### 2. Content Security Policy

Add CSP headers to allow WebGL:

```html
<meta http-equiv="Content-Security-Policy" 
      content="default-src 'self'; 
               script-src 'self' 'unsafe-eval'; 
               worker-src blob:;">
```

### 3. Liveness Detection

Prevent photo/video spoofing (basic):

```javascript
// Check for multiple frames with slight differences
const frames = [];
for (let i = 0; i < 5; i++) {
  const embedding = await extractEmbedding(video);
  frames.push(embedding);
  await sleep(200);
}

// Verify embeddings are slightly different (person is alive)
// If all identical, might be a static photo
```

---

## Browser Compatibility

| Browser | Version | Support |
|---------|---------|---------|
| Chrome  | 90+     | ✅ Full |
| Firefox | 88+     | ✅ Full |
| Safari  | 14+     | ✅ Full |
| Edge    | 90+     | ✅ Full |
| Opera   | 76+     | ✅ Full |

**Requirements:**
- WebGL 1.0 or 2.0
- getUserMedia API
- ES6+ JavaScript

---

## Troubleshooting

### Issue: Model file 404 error

**Solution:** Ensure model files are in `public/models/face_recognition/`:

```bash
# Check files exist
ls public/models/face_recognition/
# Should show: model.json and .bin files
```

### Issue: "WebGL not supported"

**Solution:** Enable hardware acceleration:

**Chrome:** Settings → System → "Use hardware acceleration when available"

**Firefox:** about:config → webgl.disabled → false

### Issue: Slow performance

**Solutions:**
1. Reduce webcam resolution
2. Increase `VERIFICATION_INTERVAL_MS`
3. Close other tabs/applications
4. Check browser console for errors

---

## Examples

See `examples/` directory for complete integration examples:

- `examples/login-flow/` - Face authentication for login
- `examples/continuous-monitoring/` - Real-time monitoring
- `examples/photo-verification/` - Verify photo vs live person

---

## Support

For issues or questions:
- Open an issue on GitHub
- Check documentation at `/docs`
- Review browser console for error messages
