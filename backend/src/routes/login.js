const express = require('express');
const router = express.Router();
const { supabase } = require('../supabaseClient');
require('dotenv').config();

// Initialize Google AI with proper error handling
const API_KEY = process.env.GOOGLE_API_KEY;
if (!API_KEY) {
    console.error('GOOGLE_AI_API_KEY is not set in environment variables');
}



router.post('/api/login', async (req, res) => {
    const { userid, password } = req.body;

    try {
        // Query the users table to find the user using username instead of user_id
        const { data: user, error } = await supabase
            .from('USERS')
            .select('*')
            .eq('username', userid)
            .eq('password', password)
            .single();

        if (error) {
            console.error('Database error:', error);
            return res.status(500).json({ success: false, message: 'Database error occurred' });
        }

        if (user) {
            console.log('Login successful');
            res.json({ 
                success: true, 
                message: 'Login successful', 
                user: { 
                    id: user.id,
                    username: user.username,
                    fname: user.fname
                } 
            });
        } else {
            console.log('Login failed');
            res.json({ success: false, message: 'Invalid username or password' });
        }
    } catch (error) {
        console.error('Error during login:', error);
        res.status(500).json({ success: false, message: 'An error occurred during login' });
    }
});

module.exports = router;