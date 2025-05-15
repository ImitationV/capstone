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


// Function to calculate current balance from TRANSACTIONS table
async function fetchCurrentBalance(userId) {
  try {
    // Fetch all transactions for the given user with type 'income' or 'expense'
    const { data, error } = await supabase
      .from('TRANSACTIONS')
      .select('amount, transaction_type')
      .eq('user_id', userId);

    if (error) {
      console.error('Supabase Error:', error.message);
      console.error('Error details:', error);
      return;
    }

    if (!data || data.length === 0) {
      console.log('No transactions found in the table for user', userId);
      return 0;
    }

    // Sum up income and expense
    let income = 0;
    let expense = 0;
    data.forEach(tx => {
      if (tx.transaction_type === 'income') {
        income += tx.amount;
      } else if (tx.transaction_type === 'expense') {
        expense += tx.amount;
      }
    });
    const balance = income - expense;
    return balance;
  } catch (err) {
    console.error('Unexpected error:', err);
    throw err;
  }
}

module.exports = {
  supabase,
  fetchUsers,
  fetchCurrentBalance
  //dbService: new DatabaseService()
};

//module.exports = new DatabaseService();