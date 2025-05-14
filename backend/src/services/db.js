require('dotenv').config();
const { createClient } = require('@supabase/supabase-js');
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_KEY; 

const supabase = createClient(supabaseUrl, supabaseKey);




// Function to fetch users - this is a test function to check if the connection to the database is working
async function fetchUsers() {
  try {
    console.log('Attempting to fetch users...');
    const { data, error } = await supabase
      .from('USERS')
      .select('*');
    
    if (error) {
      console.error('Supabase Error:', error.message);
      console.error('Error details:', error);
      return;
    }
    
    if (!data || data.length === 0) {
      console.log('No data found in the table');
      return;
    }
    
    console.log('Query successful! Number of records:', data.length);
    console.log('First record:', data[0]);
    return data;
  } catch (err) {
    console.error('Unexpected error:', err);
    throw err;
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
};