import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom'; 
import '../styles/overview.css';
import ChatbotPopup from '../components/ChatbotPopup';
import BalanceOverTimeChart from '../components/BalanceOverTimeChart'; 
import SpendingByCategoryChart from '../components/SpendingByCategoryChart'; 

function OverviewPage() {
    const navigate = useNavigate(); 
    // state to hold the user ID and loading state
    const [userId, setUserId] = useState(null);
    const [loadingUser, setLoadingUser] = useState(true); 

     // state to hold balance and savings
    const [currentBalance, setCurrentBalance] = useState('$0.00');
    const [totalSavings, setTotalSavings] = useState('$0.00');

    // read user data from localStorage
    useEffect(() => {
        const storedUser = localStorage.getItem('user');
        if (storedUser) {
            try {
                const user = JSON.parse(storedUser);
                if (user && user.id) {
                    setUserId(user.id);
                    console.log('User ID retrieved from localStorage:', user.id);
                    // fetch balance and savings using userId
                    fetchUserSummary(user.id); // fetch summary
                } else {
                    console.error('User data in localStorage is missing ID. Redirecting to login.');
                    localStorage.removeItem('user');
                    navigate('/login');
                }
            } catch (error) {
                console.error('Error parsing user data from localStorage:', error);
                 localStorage.removeItem('user'); 
                 navigate('/login');
            } finally {
                setLoadingUser(false);
            }
        } else {
            console.log('No user data found in localStorage. Redirecting to login.');
            setLoadingUser(false);
            navigate('/login'); 
        }
    }, [navigate]);

    const fetchUserSummary = async (userId) => {
        try {
            const response = await fetch(`http://localhost:4000/api/balance?userId=${userId}`);
            const result = await response.json();
            if (result.success) {
                // Format as currency
                const formattedBalance = result.balance.toLocaleString('en-US', { style: 'currency', currency: 'USD' });
                setCurrentBalance(formattedBalance);
            } else {
                setCurrentBalance('$0.00');
            }
        } catch (error) {
            console.error('Error fetching balance:', error);
            setCurrentBalance('$0.00');
        }
        setTotalSavings('$50000.00'); // Placeholder, update as needed
    };


    return (
        <>
            <div className="overview-page-content">
                <div className="header">
                    <h1>Overview</h1>
                </div>
                {/* Balance and Savings Section */}
                <div className="balance-section">
                    <div className="balance-box">
                        <h3>Current Balance</h3>
                        <p>{currentBalance}</p> 
                    </div>
                    <div className="savings-box">
                        <h3>Total Savings</h3>
                        <p>{totalSavings}</p>
                    </div>
                    <div className="menu-box">
                        <h3>Goal</h3>
                        <ul>
                            <li><p>Amount: </p></li>  
                            <li><p>Status: Active</p></li> 
                        </ul>
                    </div>
                    <div className="menu-box">
                        <h3>Quick Links</h3>
                        <ul>
                            <li><Link to="/transactions">Add Transactions</Link></li>
                            <li><Link to="/goals">Manage Goals</Link></li>
                            <li><Link to="/stock-analyzer">Stock Analyzer</Link></li>
                        </ul>
                    </div>
                </div>

                {/* Charts Section */}
                <div className="charts-section">
                    {loadingUser ? (
                        <p>Loading user data...</p>
                    ) : userId ? (
                         // Renders charts only if userId is available and not loading
                        <>
                            <div className="chart-box">
                                <BalanceOverTimeChart userId={userId} />
                            </div>
                            <div className="chart-box">
                                <SpendingByCategoryChart userId={userId} />
                                <Link to="/transactions" className="view-all-link">View all Transactions</Link>
                            </div>
                        </>
                    ) : (
                        <p>Please log in to view your financial overview.</p>
                    )}
                </div>
            </div>
            <ChatbotPopup />
        </>
    );
}

export default OverviewPage;
