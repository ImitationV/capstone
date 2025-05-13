require('dotenv').config();
const { supabase } = require('../supabaseClient');

class DatabaseService {
    // Regular user verification
    async verifyUser(userid, password) {
        try {
            const { data, error } = await supabase
                .from('USERS')
                .select('*')
                .eq('username', userid)
                .single();

            if (error) {
                console.error('Database error:', error);
                throw error;
            }

            if (!data) {
                console.log('No user found with username:', userid);
                return null;
            }

            // Verify password (implement your password hashing logic)
            const isValid = await this.verifyPassword(password, data.password);
            return isValid ? data : null;
        } catch (error) {
            console.error('User verification error:', error);
            throw error;
        }
    }

    // Get user by email (for Google auth)
    async getUserByEmail(email) {
        try {
            const { data, error } = await supabase
                .from('USERS')
                .select('*')
                .eq('email', email)
                .single();

            if (error) {
                console.error('Database error:', error);
                throw error;
            }

            return data;
        } catch (error) {
            console.error('Get user by email error:', error);
            throw error;
        }
    }

    // Create new user from Google auth
    async createGoogleUser({ email, full_name, google_id }) {
        try {
            const { data, error } = await supabase
                .from('USERS')
                .insert([
                    {
                        email,
                        full_name,
                        google_id,
                        auth_type: 'google',
                        created_at: new Date().toISOString()
                    }
                ])
                .select()
                .single();

            if (error) {
                console.error('Database error:', error);
                throw error;
            }

            return data;
        } catch (error) {
            console.error('Create Google user error:', error);
            throw error;
        }
    }

    // Helper method for password verification
    async verifyPassword(inputPassword, storedPassword) {
        // TODO: Implement proper password hashing
        return inputPassword === storedPassword;
    }
}

module.exports = new DatabaseService();