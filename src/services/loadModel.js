import tf from '@tensorflow/tfjs-node';

async function loadModel() {
    return await tf.loadGraphModel(process.env.MODEL_URL);
}

export default loadModel;
