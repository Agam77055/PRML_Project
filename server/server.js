import express from 'express';
import axios from 'axios';
import cors from 'cors';

const app = express();
app.use(cors());
app.use(express.json());

const CLOUD_RUN_URL = process.env.CLOUD_RUN_URL || 'https://chatbot-app-383089853912.asia-south1.run.app/chat';

app.get('/', async (req, res) => {
  res.status(200).send({
    message: 'Hello from ChatBot',
  });
});

app.post('/', async (req, res) => {
  try {
    const prompt = req.body.prompt;

    if (!prompt) {
        return res.status(400).send({ error: 'Prompt is required' });
    }

    const response = await axios.post(CLOUD_RUN_URL, { user_input: prompt });
    
    console.log(`Received prompt: ${prompt}`); // Log received prompt

    
    res.status(200).send({
      bot: response.data.response,
    });
  } catch (error) {
    console.error('Error calling Cloud Run service:', error);
    res.status(500).send({ error: error.message });
  }
});

const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
});