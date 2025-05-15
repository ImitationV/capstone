const express = require('express');
const router = express.Router();
const { supabase } = require('../supabaseClient');
const db = require('../services/db');
require('dotenv').config();
const { fetchCurrentBalance } = require('../services/db');

// Initialize Google AI with proper error handling
const API_KEY = process.env.GOOGLE_API_KEY;
if (!API_KEY) {
    console.error('GOOGLE_AI_API_KEY is not set in environment variables');
}

// Regular login endpoint
router.post('/api/login', async (req, res) => {
    try {
        const { userid, password } = req.body;
        // Verify user credentials
        const user = await db.verifyUser(userid, password);
        
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
            res.status(401).json({ 
                success: false, 
                message: 'Invalid credentials' 
            });
        }
    } catch (error) {
        console.error('Login error:', error);
        res.status(500).json({ 
            success: false, 
            message: 'Internal server error' 
        });
    }
});

// Verify Google token endpoint
router.post('/api/verify-google-token', async (req, res) => {
    try {
        const { token } = req.body;
        const { data: { user }, error } = await supabase.auth.getUser(token);
        
        if (error) throw error;

        if (user) {
            // Check if user exists in our database
            const dbUser = await db.getUserByEmail(user.email);
            
            if (!dbUser) {
                // Create new user if doesn't exist
                const newUser = await db.createGoogleUser({
                    email: user.email,
                    full_name: user.user_metadata.full_name,
                    google_id: user.id
                });
                return res.json({ 
                    success: true, 
                    message: 'Google login successful',
                    user: {
                        id: newUser.id,
                        email: newUser.email,
                        fname: newUser.full_name,
                        isGoogleUser: true
                    }
                });
            }

            return res.json({ 
                success: true, 
                message: 'Google login successful',
                user: {
                    id: dbUser.id,
                    email: dbUser.email,
                    fname: dbUser.full_name,
                    isGoogleUser: true
                }
            });
        }

        res.status(401).json({ 
            success: false, 
            message: 'Invalid token' 
        });
    } catch (error) {
        console.error('Token verification error:', error);
        res.status(500).json({ 
            success: false, 
            message: 'Internal server error' 
        });
    }
});

// Get user session endpoint
router.get('/api/session', async (req, res) => {
    try {
        const { token } = req.headers;
        if (!token) {
            return res.status(401).json({
                success: false,
                message: 'No token provided'
            });
        }

        const { data: { user }, error } = await supabase.auth.getUser(token);
        
        if (error) throw error;

        if (user) {
            const dbUser = await db.getUserByEmail(user.email);
            return res.json({
                success: true,
                user: {
                    id: dbUser?.id || user.id,
                    email: user.email,
                    fname: dbUser?.full_name || user.user_metadata.full_name,
                    isGoogleUser: true
                }
            });
        }

        res.status(401).json({
            success: false,
            message: 'Invalid session'
        });
    } catch (error) {
        console.error('Session verification error:', error);
        res.status(500).json({
            success: false,
            message: 'Internal server error'
        });
    }
});

router.post('/api/register', async (req, res) => {
    const { email, fname, lname, username, password } = req.body;
    try {
        // Check if username or email already exists
        const { data: existingUser, error: checkError } = await supabase
            .from('USERS')
            .select('*')
            .or(`username.eq.${username},email.eq.${email}`)
            .maybeSingle();
        if (checkError) {
            console.error('Database error:', checkError);
            return res.status(500).json({ success: false, message: 'Database error occurred' });
        }
        if (existingUser) {
            return res.status(400).json({ success: false, message: 'Username or email already exists.' });
        }
        // Insert new user
        const { data, error } = await supabase
            .from('USERS')
            .insert([{ email, fname, lname, username, password, created_at: new Date().toISOString() }])
            .select()
            .single();
        if (error) {
            console.error('Database error:', error);
            return res.status(500).json({ success: false, message: 'Database error occurred' });
        }
        res.json({ success: true, user: { id: data.id, username: data.username, fname: data.fname } });
    } catch (error) {
        console.error('Error during registration:', error);
        res.status(500).json({ success: false, message: 'An error occurred during registration' });
    }
});

// Endpoint to get current balance for a user
router.get('/api/balance', async (req, res) => {
    const userId = req.query.userId;
    if (!userId) {
        return res.status(400).json({ success: false, message: 'Missing userId parameter' });
    }
    try {
        const balance = await fetchCurrentBalance(userId);
        res.json({ success: true, balance });
    } catch (error) {
        console.error('Error fetching balance:', error);
        res.status(500).json({ success: false, message: 'Error fetching balance' });
    }
});

module.exports = router;