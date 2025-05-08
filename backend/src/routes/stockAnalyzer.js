const express = require('express');
const router = express.Router();
const axios = require('axios');

// Python service URL
const PYTHON_SERVICE_URL = 'http://localhost:8000';

router.post('/analyze', async (req, res) => {
    try {
        const { ticker, riskTolerance, ownsStock } = req.body;
        
        if (!ticker) {
            return res.status(400).json({
                error: 'Invalid request',
                details: 'Ticker symbol is required'
            });
        }

        console.log('Sending request to Python service:', {
            ticker,
            risk_tolerance: riskTolerance || 'moderate',
            owns_stock: ownsStock || false
        });
        
        // Call Python service
        const response = await axios.post(`${PYTHON_SERVICE_URL}/analyze`, {
            ticker,
            risk_tolerance: riskTolerance || 'moderate',
            owns_stock: ownsStock || false
        });
        
        console.log('Received response from Python service:', response.data);
        res.json(response.data);
    } catch (error) {
        console.error('Error analyzing stock:', error.response?.data || error.message);
        res.status(500).json({
            error: 'Failed to analyze stock',
            details: error.response?.data?.detail || error.message
        });
    }
});

module.exports = router; 