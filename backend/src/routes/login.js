const express = require('express');
const router = express.Router();
const { createClient } = require('@supabase/supabase-js');
require('dotenv').config();

// Initialize Google AI with proper error handling
const API_KEY = process.env.GOOGLE_API_KEY;
if (!API_KEY) {
    console.error('GOOGLE_AI_API_KEY is not set in environment variables');
}

// Initialize Supabase client
const supabaseUrl = 'https://idwneflrvwwcwkjlwbkz.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlkd25lZmxydnd3Y3dramx3Ymt6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1ODM1NjcsImV4cCI6MjA1NzE1OTU2N30.RmyMAOfIS1h30ne2E4AT1RB-XWpjA2DN0Bo4FW-9bmQ';
const supabase = createClient(supabaseUrl, supabaseKey);

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