import * as tf from '@tensorflow/tfjs-node';

// Function to extract dominant colors using K-means clustering
export async function extractDominantColors(predMask, imgBuffer, k = 3) {
    try {
        const img = await tf.node.decodeImage(imgBuffer); // Load image directly with TensorFlow.js
        const pixelsTensor = img.toFloat(); // Convert image to float32 tensor

        // Perform K-means clustering
        const kmeans = await tf.kmeans(pixelsTensor, k);

        // Get cluster indices and centroids
        const clusters = kmeans.clusters.arraySync();
        const centroids = kmeans.centers.arraySync();

        const segmentedAreas = {};
        clusters.forEach((clusterIndex, i) => {
            const color = centroids[clusterIndex]; // Extract centroid RGB values
            const maskData = predMask.reshape([256, 256]).equal(tf.scalar(clusterIndex)).toInt(); // Create mask
            const mask = maskData.arraySync();
            const segmentedArea = applyMask(img, mask); // Apply mask to original image
            segmentedAreas[clusterIndex] = { color, segmentedArea };
        });

        return segmentedAreas;
    } catch (error) {
        console.error('Error extracting dominant colors:', error);
        throw error;
    }
}

// Apply mask to image and extract segmented area
function applyMask(imgTensor, mask) {
    const segmentedArea = tf.tidy(() => {
        const maskedImg = imgTensor.mul(tf.tensor3d(mask)); // Apply mask to image tensor

        // Convert masked image tensor back to canvas-compatible format (data URL)
        const canvasImg = tf.node.encodeJpeg(maskedImg); // Encode masked image as JPEG

        // Convert JPEG tensor to data URL
        const dataUrl = canvasImg.arraySync(); // Convert to array
        return dataUrl;
    });

    return segmentedArea;
}

// Function to determine the user's palette using SCA principles
export function determinePalette(dominantColors) {
    const peach = [255, 229, 180];
    const purple = [128, 0, 128];

    const lipsColor = dominantColors[1]?.color || [0, 0, 0]; // Default to black if not found
    const skinColor = dominantColors[4]?.color || [0, 0, 0]; // Default to black if not found
    const hairColor = dominantColors[5]?.color || [0, 0, 0]; // Default to black if not found
    const eyesColor = dominantColors[2]?.color || [0, 0, 0]; // Default to black if not found

    // Hue determination
    const hue = colorDistance(lipsColor, peach) < colorDistance(lipsColor, purple) ? 'warm' : 'cool';

    // Saturation determination
    const skinSaturation = Math.sqrt(
        Math.pow(skinColor[0] - tf.mean(skinColor), 2) +
        Math.pow(skinColor[1] - tf.mean(skinColor), 2) +
        Math.pow(skinColor[2] - tf.mean(skinColor), 2)
    );
    const saturationThreshold = 20; // Set a threshold value
    const saturation = skinSaturation > saturationThreshold ? 'bright' : 'muted';

    // Value determination
    const valueThreshold = 127; // Set a threshold value
    const meanBrightness = tf.mean([tf.mean(skinColor), tf.mean(hairColor), tf.mean(eyesColor)]);
    const value = meanBrightness > valueThreshold ? 'light' : 'dark';

    // Contrast determination
    const contrastThreshold = 50; // Set a threshold value
    const contrast = Math.abs(tf.mean(hairColor) - tf.mean(eyesColor)) > contrastThreshold ? 'high' : 'low';

    // Create a metric vector
    const metricVector = [hue === 'warm', saturation === 'bright', value === 'light', contrast === 'high'];

    // Define seasonal palettes
    const palettes = {
        spring: [true, true, true, true],
        summer: [false, true, true, false],
        autumn: [true, false, false, true],
        winter: [false, false, false, false]
    };

    // Determine the closest palette
    let minDistance = Number.MAX_SAFE_INTEGER;
    let bestPalette = null;

    for (const [season, paletteVector] of Object.entries(palettes)) {
        const distance = metricVector.reduce((acc, val, idx) => acc + (val !== paletteVector[idx] ? 1 : 0), 0);
        if (distance < minDistance) {
            minDistance = distance;
            bestPalette = season;
        }
    }

    return bestPalette;
}

// Helper function to calculate Euclidean distance between two colors
function colorDistance(color1, color2) {
    const [r1, g1, b1] = color1;
    const [r2, g2, b2] = color2;
    return Math.sqrt(Math.pow(r1 - r2, 2) + Math.pow(g1 - g2, 2) + Math.pow(b1 - b2, 2));
}
