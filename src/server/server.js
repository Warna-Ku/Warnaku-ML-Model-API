import express from 'express';
import cors from 'cors';
import tf from '@tensorflow/tfjs-node';
import multer from 'multer';
import sharp from 'sharp';
import moment from 'moment-timezone';

const app = express();
const PORT = process.env.PORT || 3000;

app.use(express.json());
app.use(cors());

app.get('/', (req, res) => {
    res.send('<h1>Server is running...</h1>');
});

app.listen(PORT, '0.0.0.0', () => 
    console.log(`listening on port ${PORT}`)
);
