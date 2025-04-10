import express from 'express';
import cors from 'cors';
import { spawn } from 'child_process';

const app = express();
app.use(cors());
app.use(express.json());

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

    console.log(`Received prompt: ${prompt}`); // Log received prompt

    const pythonProcess = spawn('python', ['/Users/agamsingh/Desktop/CSL2050/Project/Chatbot/college_chatbot.py', prompt]);

    let result = '';
    
    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });
    

    pythonProcess.stderr.on('data', (data) => {
      console.error(`stderr: ${data}`);
    });

    pythonProcess.on('close', (code) => {
      console.log(`Python process exited with code ${code}`);
      res.status(200).send({
        bot: result.trim(),
      });
    });
  } catch (error) {
    console.error(error);
    res.status(500).send({ error: error.message });
  }
});

app.listen(5001, () =>
  console.log('Server is running on port http://localhost:5001')
);
