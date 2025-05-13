require('dotenv').config();
const { supabase } = require('../supabaseClient');



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

module.exports = {
  supabase,
  fetchUsers
};