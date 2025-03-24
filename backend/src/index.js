const express = require('express');

// Import cors middleware to enable Cross-Origin Resource Sharing (CORS)
// This allows our frontend application to make requests to our backend API
const cors = require('cors');
const { supabase, fetchUsers } = require('./services/db');
const loginRouter = require('./routes/login');
const app = express();


// This allows the frontend to make requests to the backend
const corsOptions = {
    origin: 'http://localhost:5173',
};

// Middlewares
app.use(cors(corsOptions));  // allows cross-origin requests from your React app
app.use(express.json());    // parses JSON bodies in requests



// Simple test route (health check)
app.get('/api/health', (req, res) => {
  res.json({ status: 'OK', message: 'Backend is running!' });
});

// Call the function to check user data
// fetchUsers();

// This allows the frontend to send login requests to the backend
app.post('/api/login', loginRouter);


// Start the server
const PORT = process.env.PORT || 4000;
app.listen(PORT, () => {
    console.log('Navigating to http://localhost:4000/api/health to make see the success message');
    console.log(`Server listening on port ${PORT}`);
});