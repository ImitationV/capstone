import React, { useState } from 'react';
import '../styles/transactions.css';
import ChatbotPopup from './ChatbotPopup';


function TransactionsPage() {
  const [transactionType, setTransactionType] = useState('');
  const [expenseOption, setExpenseOption] = useState('');
  const [savingsOption, setSavingsOption] = useState('');
  const [error, setError] = useState('');

  const handleTransactionTypeChange = (event) => {
    setTransactionType(event.target.value);
    setExpenseOption('');
    setSavingsOption('');
    setError('');
  };

  const handleExpenseChange = (event) => {
    setExpenseOption(event.target.value);
    setError('');
  };

  const handleSavingsChange = (event) => {
    setSavingsOption(event.target.value);
    setError('');
  };

  const handleSubmit = (event) => {
    event.preventDefault();

    if (!transactionType || (transactionType === 'expense' && !expenseOption) || (transactionType === 'savings' && !savingsOption)) { // Checks if all dropdowns are selected
      setError('Please select an option from all dropdowns.');
      return;
    }

    // Default values for expense and savings options set to 0 if user doesn't select any
    const expenseValue = expenseOption || 0;
    const savingsValue = savingsOption || 0;

    console.log('Transaction Type:', transactionType);
    console.log('Expense Option:', expenseValue);
    console.log('Savings Option:', savingsValue);

    // Resets form after submit
    setTransactionType('');
    setExpenseOption('');
    setSavingsOption('');
  };

  return (
    <><div className="transactions-page">
      <div className="transaction-container">
        <h1>Transaction</h1>
        <form onSubmit={handleSubmit}>
          {error && <p className="error">{error}</p>}

          <div className="form-group">
            <label htmlFor="transactionType">Choose Transaction Type</label>
            <select
              id="transactionType"
              value={transactionType}
              onChange={handleTransactionTypeChange}
              className="select-input"
            >
              <option value="">Select...</option>
              <option value="expense">Expense</option>
              <option value="savings">Savings</option>
            </select>
          </div>

          {transactionType === 'expense' && (
            <div className="form-group">
              <label htmlFor="expenseOption">Expense Options</label>
              <select
                id="expenseOption"
                value={expenseOption}
                onChange={handleExpenseChange}
                className="select-input"
              >
                <option value="">Select...</option>
                <option value="rent">Rent</option>
                <option value="loan">Loan</option>
                <option value="others">Others</option>
              </select>
            </div>
          )}

          {transactionType === 'savings' && (
            <div className="form-group">
              <label htmlFor="savingsOption">Savings Options</label>
              <select
                id="savingsOption"
                value={savingsOption}
                onChange={handleSavingsChange}
                className="select-input"
              >
                <option value="">Select...</option>
                <option value="liquid">Liquid</option>
                <option value="interest">Interest</option>
                <option value="others">Others</option>
              </select>
            </div>
          )}

          <button type="submit" className="submit-button">Submit</button>
        </form>
      </div>
    </div><ChatbotPopup /></>

  );
}

export default TransactionsPage;