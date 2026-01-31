// Plant Disease Detection - Frontend JavaScript

let cameraStream = null;
let currentMode = 'upload';
let modelReady = false;

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    setupDragDrop();
    setupFileInput();
    checkModelStatus();
});

// Check if model is loaded on backend
function checkModelStatus() {
    fetch('/health', {
        method: 'GET'
    })
    .then(response => response.json())
    .then(data => {
        if (data.model_loaded) {
            modelReady = true;
            document.getElementById('errorMessage').style.display = 'none';
        } else {
            modelReady = false;
            showError('⚠ Model is loading... Please wait or try again in a moment.');
        }
    })
    .catch(error => {
        console.log('Health check error (backend may be starting):', error);
        // Don't show error on initial load - backend might be starting
        modelReady = false;
    });
}

// Mode Switching
function switchMode(mode) {
    currentMode = mode;
    
    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById(mode + '-mode-btn').classList.add('active');
    
    // Update sections
    document.querySelectorAll('.mode-section').forEach(sec => sec.classList.remove('active'));
    document.getElementById(mode + '-section').classList.add('active');
    
    // Stop camera if switching away
    if (mode !== 'camera' && cameraStream) {
        stopCamera();
    }
}

// File Upload Handling
function setupDragDrop() {
    const dropZone = document.getElementById('dropZone');
    if (!dropZone) {
        console.error('dropZone element not found');
        return;
    }
    
    console.log('Setting up drag-drop listeners');
    
    // Remove old listeners by cloning and replacing
    const newDropZone = dropZone.cloneNode(true);
    dropZone.parentNode.replaceChild(newDropZone, dropZone);
    
    // Add click listener
    newDropZone.onclick = function(e) {
        console.log('Drop zone clicked');
        document.getElementById('fileInput').click();
    };
    
    // Add drag-drop listeners
    newDropZone.ondragover = function(e) {
        e.preventDefault();
        e.stopPropagation();
        newDropZone.classList.add('dragover');
    };
    
    newDropZone.ondragleave = function(e) {
        e.preventDefault();
        e.stopPropagation();
        newDropZone.classList.remove('dragover');
    };
    
    newDropZone.ondrop = function(e) {
        e.preventDefault();
        e.stopPropagation();
        newDropZone.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            handleFileSelect(files[0]);
        }
    };
}

function handleDropZoneClick(e) {
    console.log('Drop zone clicked');
    document.getElementById('fileInput').click();
}

function handleDragOver(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropZone').classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropZone').classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.stopPropagation();
    document.getElementById('dropZone').classList.remove('dragover');
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFileSelect(files[0]);
    }
}

function setupFileInput() {
    const fileInput = document.getElementById('fileInput');
    // Remove old listener to prevent duplicates by cloning
    const newFileInput = fileInput.cloneNode(true);
    fileInput.parentNode.replaceChild(newFileInput, fileInput);
    
    // Add new listener
    newFileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });
}

function handleFileSelect(file) {
    // Validate file
    if (!['image/jpeg', 'image/png'].includes(file.type)) {
        showError('Invalid file type. Please use JPG or PNG.');
        return;
    }
    
    if (file.size > 10 * 1024 * 1024) {
        showError('File too large. Max 10MB.');
        return;
    }
    
    // Store the file and show analyze button
    window.selectedFile = file;
    console.log('File selected:', file.name);
    
    // Show image preview
    const reader = new FileReader();
    reader.onload = function(e) {
        // Update upload area to show image preview
        const dropZone = document.getElementById('dropZone');
        dropZone.innerHTML = `<img src="${e.target.result}" style="max-width: 100%; max-height: 300px; border-radius: 8px;" alt="Preview">`;
    };
    reader.readAsDataURL(file);
    
    // Show Analyze button
    document.getElementById('analyzeBtn').style.display = 'inline-block';
    document.getElementById('errorMessage').style.display = 'none';
}

function analyzeImage() {
    if (!window.selectedFile) {
        showError('Please select an image first');
        return;
    }
    console.log('Analyzing image:', window.selectedFile.name);
    uploadImage(window.selectedFile);
}

function uploadImage(file) {
    console.log('Starting upload for file:', file.name, 'Size:', file.size);
    showLoading();
    
    const formData = new FormData();
    formData.append('file', file);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        console.log('Response status:', response.status, response.statusText);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log('Response data received:', data);
        hideLoading();
        
        if (data.status === 'success') {
            console.log('Analysis successful, displaying results...');
            console.log('Detections:', data.detections);
            console.log('Annotated image available:', !!data.annotated_image);
            displayResults(data.detections, data.annotated_image);
        } else if (data.error) {
            console.log('Backend error:', data.error);
            showError('⚠ Error: ' + data.error);
        } else {
            console.log('Unknown response format:', data);
            showError('⚠ Unexpected response from server');
        }
    })
    .catch(error => {
        console.error('Fetch error:', error);
        hideLoading();
        showError('⚠ Error analyzing image: ' + error.message);
    });
}

// Camera Handling
async function startCamera() {
    try {
        const cameraPlaceholder = document.getElementById('cameraPlaceholder');
        const cameraVideo = document.getElementById('cameraVideo');
        const startBtn = document.getElementById('startCameraBtn');
        const stopBtn = document.getElementById('stopCameraBtn');
        const captureBtn = document.getElementById('captureBtn');
        
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: 'environment' }
        });
        
        cameraVideo.srcObject = cameraStream;
        cameraPlaceholder.style.display = 'none';
        cameraVideo.style.display = 'block';
        startBtn.style.display = 'none';
        stopBtn.style.display = 'inline-block';
        captureBtn.style.display = 'inline-block';
    } catch (error) {
        showError('Cannot access camera: ' + error.message);
    }
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach(track => track.stop());
        cameraStream = null;
        
        document.getElementById('cameraPlaceholder').style.display = 'flex';
        document.getElementById('cameraVideo').style.display = 'none';
        document.getElementById('startCameraBtn').style.display = 'inline-block';
        document.getElementById('stopCameraBtn').style.display = 'none';
        document.getElementById('captureBtn').style.display = 'none';
    }
}

function captureFrame() {
    const video = document.getElementById('cameraVideo');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0);
    
    // Convert to blob and upload
    canvas.toBlob(blob => {
        const file = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
        uploadImage(file);
    }, 'image/jpeg');
}

// Results Display
function displayResults(detections, annotatedImage) {
    console.log('displayResults called with:', { detections, annotatedImage });
    
    // Set the image
    if (annotatedImage) {
        document.getElementById('resultImage').src = annotatedImage;
    }
    
    const resultsDiv = document.getElementById('detectionResults');
    resultsDiv.innerHTML = '';
    
    // Handle detections
    if (!detections || detections.length === 0) {
        resultsDiv.innerHTML = '<div class="no-detection">✓ No diseases detected! Plant is healthy.</div>';
    } else {
        console.log(`Found ${detections.length} detections`);
        detections.forEach((detection, idx) => {
            console.log(`Processing detection ${idx}:`, detection);
            
            const confidence = detection.confidence || 0;
            const confidenceLevel = confidence > 0.7 ? 'high-confidence' :
                                   confidence > 0.5 ? 'medium-confidence' :
                                   'low-confidence';
            
            const item = document.createElement('div');
            item.className = `detection-item ${confidenceLevel}`;
            
            const className = detection.class_name || detection.className || 'Unknown';
            const bbox = detection.bbox || [];
            
            item.innerHTML = `
                <div class="detection-info">
                    <h4>🔍 ${className}</h4>
                    <p>Confidence: ${(confidence * 100).toFixed(1)}%</p>
                    ${bbox.length > 0 ? `<p>Location: [${bbox.join(', ')}]</p>` : ''}
                </div>
                <span class="confidence-score">${(confidence * 100).toFixed(1)}%</span>
            `;
            resultsDiv.appendChild(item);
        });
    }
    
    // Show results section
    const resultsSection = document.getElementById('resultsSection');
    if (resultsSection) {
        resultsSection.style.display = 'block';
        document.getElementById('errorMessage').style.display = 'none';
        
        // Scroll to results
        setTimeout(() => {
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }, 100);
    }
}

// UI Helpers
function showLoading() {
    document.getElementById('loadingSpinner').style.display = 'flex';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('errorMessage').style.display = 'none';
}

function hideLoading() {
    document.getElementById('loadingSpinner').style.display = 'none';
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
    document.getElementById('resultsSection').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
}

function resetApp() {
    console.log('Resetting app...');
    
    // Keep results section visible so user can see the previous image
    // Just clear the detection results list
    const detectionResults = document.getElementById('detectionResults');
    if (detectionResults) {
        detectionResults.innerHTML = '<p style="text-align:center; color:#999;">Ready for new analysis...</p>';
    }
    
    // Clear file input for new selection
    const fileInput = document.getElementById('fileInput');
    if (fileInput) {
        fileInput.value = '';
        // Reset the input type to clear cached file
        fileInput.type = '';
        fileInput.type = 'file';
    }
    
    // Re-setup event listeners for fresh state
    setupDragDrop();
    setupFileInput();
    
    // Hide loading and error messages
    document.getElementById('errorMessage').style.display = 'none';
    document.getElementById('loadingSpinner').style.display = 'none';
    
    // Show the upload interface again
    switchMode('upload');
    
    // Scroll to upload section so user can see and use it
    const uploadSection = document.getElementById('upload-section');
    if (uploadSection) {
        uploadSection.scrollIntoView({ behavior: 'smooth' });
    }
    
    console.log('App reset complete - upload interface ready');
}
