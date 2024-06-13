import express from 'express';
import multer from 'multer';
import axios from 'axios';
import fs from 'fs';
import FormData from 'form-data';

const app = express();
const upload = multer({ dest: 'uploads/' });
const PORT = 3000;
const FLASK_API_URL = 'http://localhost:5000/predict';

// Endpoint to handle image upload and prediction
app.post('/predict', upload.single('image'), async (req, res) => {
    try {
        const imagePath = req.file.path;
        const form = new FormData();
        form.append('image', fs.createReadStream(imagePath));

        const response = await axios.post(FLASK_API_URL, form, {
            headers: {
                ...form.getHeaders()
            }
        });

        fs.unlinkSync(imagePath);

        res.json(response.data);
    } catch (error) {
        res.status(500).send(error.message);
    }
});

// Start the server
app.listen(PORT, () => {
    console.log(`Server is running on http://localhost:${PORT}`);
});
