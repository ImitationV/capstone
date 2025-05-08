const express = require('express');
const router = express.Router();
const { GoogleGenerativeAI } = require('@google/generative-ai');
require('dotenv').config();

// Initialize Google AI with proper error handling
const API_KEY = process.env.GOOGLE_API_KEY;
if (!API_KEY) {
    console.error('GOOGLE_AI_API_KEY is not set in environment variables');
}
const genAI = new GoogleGenerativeAI(API_KEY);

// System prompt for the financial advisor
const SYSTEM_PROMPT = `You are a helpful financial advisor. Provide clear and actionable financial guidance 
in response to user questions. Keep answers concise, practical, and easy to understand.
Avoid legal disclaimers or lengthy disclaimers. Try to find sources of your answer. If 
you cant find any source mention the user that it is from your assumption.`;

router.post('/', async (req, res) => {
    try {
        const { message } = req.body;
        
        if (!message) {
            return res.status(400).json({ error: 'Message is required' });
        }

        // Get the model
        const model = genAI.getGenerativeModel({ model: "gemini-2.0-flash-001" });

        // Create chat session
        const chat = model.startChat({
            history: [
                {
                    role: "user",
                    parts: [{ text: SYSTEM_PROMPT }],
                },
                {
                    role: "model",
                    parts: [{ text: "I understand. I will provide clear and actionable financial advice." }],
                },
            ],
            generationConfig: {
                maxOutputTokens: 2048,
            },
        });

        // Send message and get response
        const result = await chat.sendMessage(message);
        const response = await result.response;
        const text = response.text();

        res.json({ response: text });
    } catch (error) {
        console.error('Chat error:', error);
        // Send a more detailed error response
        res.status(500).json({ 
            error: 'An error occurred while processing your request',
            details: error.message,
            apiKeySet: !!API_KEY // This will help debug if the API key is set
        });
    }
});

module.exports = router; 