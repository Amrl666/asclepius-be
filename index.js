const Hapi = require('@hapi/hapi');
const { loadModel, predict } = require('./ml'); // Import the model loading and prediction functions
const multer = require('multer');
const admin = require('firebase-admin');

// Initialize Firebase Admin SDK
const serviceAccount = require('./key.json'); // Ensure this file is in the same directory

admin.initializeApp({
    credential: admin.credential.cert(serviceAccount)
});

const db = admin.firestore();

// Set up multer for file uploads
const upload = multer({
    limits: { fileSize: 1000000 }, // 1MB limit
    storage: multer.memoryStorage() // Store files in memory
});

// Create Hapi server
const init = async () => {
    // Load the machine learning model
    const model = await loadModel();
    console.log('Model loaded!');

    const server = Hapi.server({
        port: process.env.PORT || 3000,
        host: '0.0.0.0',
    });

    // Error handling middleware
    server.ext('onPreResponse', (request, h) => {
        const response = request.response;

        // Check if the response is an error
        if (response.isBoom) {
            console.error(response); // Log the error
            return h.response({
                status: 'error',
                message: response.message,
                data: response.output.payload
            }).code(response.output.statusCode);
        }

        return h.continue;
    });

    // Define the /predict route
    server.route({
        method: 'POST',
        path: '/predict',
        options: {
            payload: {
                output: 'stream',
                parse: true,
                multipart: true
            }
        },
        handler: async (request) => {
            // Get the uploaded image
            const { image } = request.payload;

            // Log the uploaded image object
            console.log('Uploaded image:', image);

            // Check if the image was provided
            if (!image) {
                return {
                    status: 'fail',
                    message: 'Image not provided'
                };
            }

            // Validate the image format
            if (!image.hapi || !image.hapi.mimetype || !image.hapi.mimetype.startsWith('image/')) {
                return {
                    status: 'fail',
                    message: 'Invalid image format. Please upload a valid image.'
                };
            }

            try {
                // Read the image data from the stream
                const chunks = [];
                image.on('data', (chunk) => {
                    chunks.push(chunk);
                });

                image.on('end', async () => {
                    const buffer = Buffer.concat(chunks); // Combine all chunks into a single buffer

                    try {
                        // Get prediction result by passing the model and image buffer
                        const predictions = await predict(model, buffer);
                        const result = predictions[0] > 0.5 ? 'Cancer' : 'Non-cancer'; // Threshold for classification

                        // Prepare response
                        const suggestion = result === 'Cancer' ? 'Segera periksa ke dokter!' : 'Penyakit kanker tidak terdeteksi.';
                        const id = Math.random().toString(36).substring(2); // Generate a random ID
                        const createdAt = new Date().toISOString();

                        // Save prediction result to Firestore
                        await db.collection('predictions').doc(id).set({
                            result,
                            suggestion,
                            createdAt
                        });

                        // Return prediction result
                        return {
                            status: 'success',
                            message: 'Model is predicted successfully',
                            data: {
                                id,
                                result,
                                suggestion,
                                createdAt
                            }
                        };
                    } catch (predictionError) {
                        console.error('Error during prediction:', predictionError);
                        return {
                            status: 'error',
                            message: 'An error occurred while processing your request.'
                        };
                    }
                });

                image.on('error', (err) => {
                    console.error('Error reading the image stream:', err);
                    return {
                        status: 'fail',
                        message: 'Error processing the image'
                    };
                });
            } catch (error) {
                console.error('Unexpected error:', error);
                return {
                    status: 'error',
                    message: 'An unexpected error occurred.'
                };
            }
        }
    });

    // Additional route for health check
    server.route({
        method: 'GET',
        path: '/health',
        handler: (request, h) => {
            return {
                status: 'success',
                message: 'Server is running smoothly'
            };
        }
    });

    // Middleware for logging requests
    server.ext('onPreHandler', (request, h) => {
        console.log(`Request received: ${request.method.toUpperCase()} ${request.path}`);
        return h.continue;
    });

    // Start the server
    await server.start();
    console.log(`Server running on ${server.info.uri}`);
};

// Handle process termination
process.on('unhandledRejection', (err) => {
    console.log(err);
    process.exit(1);
});

// Initialize the server
init(); 
