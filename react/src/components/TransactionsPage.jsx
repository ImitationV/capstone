import React, { useState, useEffect } from 'react';
import '../styles/transactions.css'; 
import ChatbotPopup from './ChatbotPopup';
import { supabase } from '../../supabaseClient';





function TransactionsPage() {
    // State variables for form inputs and messages
    const [transactionType, settransactionType] = useState('Expense');
    const [transactionDate, setTransactionDate] = useState('');
    const [transactionTime, setTransactionTime] = useState('');
    const [category, setCategory] = useState('');
    const [amount, setAmount] = useState('');
    const [paymentMode, setPaymentMode] = useState('');
    const [error, setError] = useState('');
    const [successMessage, setSuccessMessage] = useState('');
    const [currentUserId, setCurrentUserId] = useState(null); // State to store the logged-in user ID
    const [currentBalance, setCurrentBalance] = useState(null);

    // Effect to get the user ID from localStorage
    useEffect(() => {
        const user = JSON.parse(localStorage.getItem('user'));
        if (user && user.id) {
            setCurrentUserId(user.id);
            fetchCurrentBalance(user.id);
        } else {
            // Handle case where user is not logged in or user data is incomplete
            setError('User not logged in. Please log in to add transactions.');
        }
    }, []); 

    // Handlers for form input changes
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

    // Function to handle form submission
    const handleSubmit = async (event) => {
        event.preventDefault();
        setSuccessMessage('');
        setError('');

        // Check if user ID is available
        if (!currentUserId) {
            setError('User ID not found. Please log in.');
            return;
        }

        if (!transactionDate || !transactionTime || !category || !amount || !paymentMode) {
            setError('Please fill in all the required fields.');
            return;
        }

        // Validate amount is a positive number
        const parsedAmount = parseFloat(amount);
        if (isNaN(parsedAmount) || parsedAmount <= 0) {
             setError('Please enter a valid positive amount.');
             return;
        }

        // Format the date and time
        const transactionData = {
            time_created: transactionTime, // Stores time
            date_created: transactionDate, // Stores date
            transaction_type: transactionType.toLowerCase(),
            category: category, // Stores selected category
            amount: parsedAmount, // Stores the validated amount
            payment_mode: paymentMode.toLowerCase(), // Stores payment method
            user_id: currentUserId, // Use the fetched user ID
        };

        try {
            const { data, error: insertError } = await supabase
                .from('TRANSACTIONS')
                .insert([transactionData]);

            if (insertError) {
                console.error('Error inserting data:', insertError);
                setError(`Failed to save transaction: ${insertError.message}`);
            } else {
                console.log('Transaction saved:', data);
                setSuccessMessage('Transaction saved successfully!');
                // Reset form fields after successful submission
                settransactionType('Expense');
                setTransactionDate('');
                setTransactionTime('');
                setCategory('');
                setAmount('');
                setPaymentMode('');
                setError(''); // Clear any previous errors
                fetchCurrentBalance(currentUserId);
            }
        } catch (error) {
            console.error('An unexpected error occurred:', error);
            setError('An unexpected error occurred while saving the transaction.');
        }
    };

    const fetchCurrentBalance = async (userId) => {
        try {
            const response = await fetch(`/api/current_balance?user_id=${userId}`);
            const result = await response.json();
            if (result.success) {
                setCurrentBalance(result.balance);
                // Optionally update localStorage or global state here
            }
        } catch (err) {
            console.error('Failed to fetch current balance:', err);
        }
    };

    return (
        <div className="transactions-page">
            <div className="transaction-container">
                <h2>New Transaction</h2>
                <form onSubmit={handleSubmit}>
                    {error && <p className="error-message">{error}</p>} 
                    {successMessage && <p className="success-message">{successMessage}</p>}

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
                            {transactionType === 'Expense' ? (
                                <>
                                    <option value="food">Food</option>
                                    <option value="transportation">Transportation</option>
                                    <option value="entertainment">Entertainment</option>
                                    <option value="utilities">Utilities</option>
                                    <option value="shopping">Shopping</option>
                                    <option value="health">Health</option>
                                    <option value="education">Education</option>
                                    <option value="other">Other Expense</option>
                                </>
                            ) : (
                                <>
                                    <option value="salary">Salary</option>
                                    <option value="investment">Investment</option>
                                    <option value="gift">Gift</option>
                                    <option value="other_income">Other Income</option>
                                </>
                            )}
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
                            min="0.01"  
                            step="0.01" 
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
                             <label>
                                <input
                                    type="radio"
                                    name="paymentMode"
                                    value="online transfer"
                                    checked={paymentMode === 'online transfer'}
                                    onChange={handlePaymentModeChange}
                                    required
                                />
                                Online Transfer
                            </label>
                        </div>
                    </div>

                    <div className="button-group">
                        <button type="submit" className="create-button">Add Transaction</button>
                    </div>
                </form>
            </div>
            <ChatbotPopup />
        </div>
    );
}

export default TransactionsPage;
