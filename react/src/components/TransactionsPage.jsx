import React, { useState } from 'react';

import '../styles/transactions.css'; 
import { createClient } from '@supabase/supabase-js';
import ChatbotPopup from './ChatbotPopup';
// Initialize Supabase client 
const supabaseUrl = 'https://idwneflrvwwcwkjlwbkz.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlkd25lZmxydnd3Y3dramx3Ymt6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1ODM1NjcsImV4cCI6MjA1NzE1OTU2N30.RmyMAOfIS1h30ne2E4AT1RB-XWpjA2DN0Bo4FW-9bmQ';
const supabase = createClient(supabaseUrl, supabaseKey);

function TransactionsPage() {
  const [transactionType, settransactionType] = useState('Expense');
  const [transactionDate, setTransactionDate] = useState('');
  const [transactionTime, setTransactionTime] = useState('');
  const [category, setCategory] = useState('');
  const [amount, setAmount] = useState('');
  const [paymentMode, setPaymentMode] = useState('');
  const [error, setError] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const handletransactionTypeChange = (event) => {
    settransactionType(event.target.value);
  };

  const handleDateChange = (event) => {
    setTransactionDate(event.target.value);
  };

  const handleTimeChange = (event) => {
    setTransactionTime(event.target.value);
  };

  const handleCategoryChange = (event) => {
    setCategory(event.target.value);
  };

  const handleAmountChange = (event) => {
    setAmount(event.target.value);
  };

  const handlePaymentModeChange = (event) => {
    setPaymentMode(event.target.value);
  };
  /*
  // Random user ID generator function
  const getRandomInt = (min, max) => {
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min + 1)) + min; 
  };
  */
 
  // Function to handle form submission
  const handleSubmit = async (event) => {
    event.preventDefault();
    setSuccessMessage('');
    setError('');

    if (!transactionDate || !transactionTime || !category || !amount || !paymentMode) {
      setError('Please fill in all the required fields.');
      return;
    }

    const transactionData = {
      time_created: transactionTime, // Stores time 
      date_created: transactionDate, // Stores date 
      transaction_type: transactionType.toLowerCase(),
      category: category, // Stores selected category
      amount: parseFloat(amount), // Stores the amount
      payment_mode: paymentMode.toLowerCase().replace(' ', ' '), // Stores payment method
      //user_id: getRandomInt(900000,999999), // Generate a random user ID between 900000 and 999999
      user_id: 2, // usman's user ID
    };

    try {
      const { data, error } = await supabase
        .from('TRANSACTIONS')
        .insert([transactionData]);

      if (error) {
        console.error('Error inserting data:', error);
        setError('Failed to save transaction.');
      } else {
        console.log('Transaction saved:', data);
        setSuccessMessage('Transaction saved successfully!');
        settransactionType('Expense');
        setTransactionDate('');
        setTransactionTime('');
        setCategory('');
        setAmount('');
        setPaymentMode('');
        setError('');
      }
    } catch (error) {
      console.error('An unexpected error occurred:', error);
      setError('An unexpected error occurred.');
    }
  };

  return (
    <><div className="transactions-page">
      <div className="transaction-container">
        <h2>New Transaction</h2>
        <form onSubmit={handleSubmit}>
          {error && <p className="error">{error}</p>}
          {successMessage && <p className="success">{successMessage}</p>}

          <div className="form-group">
            <label htmlFor="transactionType">Transaction Type *</label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="transactionType"
                  value="Income"
                  checked={transactionType === 'Income'}
                  onChange={handletransactionTypeChange}
                />
                Income
              </label>
              <label>
                <input
                  type="radio"
                  name="transactionType"
                  value="Expense"
                  checked={transactionType === 'Expense'}
                  onChange={handletransactionTypeChange}
                />
                Expense
              </label>
            </div>
          </div>

          <div className="form-group">
            <label htmlFor="transactionDate">Choose a Date *</label>
            <input
              type="date"
              id="transactionDate"
              value={transactionDate}
              onChange={handleDateChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="transactionTime">Choose a Time *</label>
            <input
              type="time"
              id="transactionTime"
              value={transactionTime}
              onChange={handleTimeChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="category">Category *</label>
            <select
              id="category"
              value={category}
              onChange={handleCategoryChange}
              required
            >
              <option value="">Select Category</option>
              <option value="food">Food</option> {}
              <option value="transportation">Transportation</option>
              <option value="entertainment">Entertainment</option>
              <option value="utilities">Utilities</option>
              <option value="salary">Salary</option>
              <option value="investment">Investment</option>
              {}
            </select>
          </div>

          <div className="form-group">
            <label htmlFor="amount">Enter Amount *</label>
            <input
              type="number"
              id="amount"
              value={amount}
              onChange={handleAmountChange}
              required
            />
          </div>

          <div className="form-group">
            <label htmlFor="paymentMode">Payment Mode *</label>
            <div className="radio-group">
              <label>
                <input
                  type="radio"
                  name="paymentMode"
                  value="cash"
                  checked={paymentMode === 'cash'}
                  onChange={handlePaymentModeChange}
                  required
                />
                Cash
              </label>
              <label>
                <input
                  type="radio"
                  name="paymentMode"
                  value="debit card"
                  checked={paymentMode === 'debit card'}
                  onChange={handlePaymentModeChange}
                  required
                />
                Debit Card
              </label>
              <label>
                <input
                  type="radio"
                  name="paymentMode"
                  value="credit card"
                  checked={paymentMode === 'credit card'}
                  onChange={handlePaymentModeChange}
                  required
                />
                Credit Card
              </label>
            </div>
          </div>

          <div className="button-group">
            <button type="submit" className="create-button">Add</button>
          </div>
        </form>
      </div>
    </div><ChatbotPopup /></>

  );
}

export default TransactionsPage;