const tfjs = require('@tensorflow/tfjs-node');

async function loadModel() {
    const modelUrl = "https://storage.googleapis.com/bkt-amirul/model/model.json"; // Ganti dengan URL model Anda
    return await tfjs.loadGraphModel(modelUrl);
}

function predict(model, imageBuffer) {
    const tensor = tfjs.node
        .decodeImage(imageBuffer, 3) // Decode as RGB
        .resizeBilinear([224, 224]) // Resize to model input size
        .expandDims(0) // Add batch dimension
        .toFloat()
        .div(tfjs.scalar(255)); // Normalize the image to [0, 1]

    return model.predict(tensor).data();
}

module.exports = { loadModel, predict };