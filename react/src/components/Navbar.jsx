import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom'; 

import '../styles/navbar.css';

function Navbar() {
    const navigate = useNavigate();
    const location = useLocation(); // gets current location
    const [fname, setFname] = useState('');

    useEffect(() => {
        // Get user information from localStorage
        const user = JSON.parse(localStorage.getItem('user'));
        if (user) {
            setFname(user.fname);
        }
    }, []);

    const goToLogin = () => {
        // Clear user data from localStorage
        localStorage.removeItem('user');
        navigate('/');
    };

    return (
        <div className="sidebar">
            <div className="sidebar-header">
                <h2>Budget Buddy</h2>
            </div>
            <nav>
                <ul>
                    <li><a href="/overview" className={location.pathname === '/overview' ? 'active' : ''}>Overview</a></li>
                    <li><a href="/transactions" className={location.pathname === '/transactions' ? 'active' : ''}>Transactions</a></li>
                    <li><a href="/goals" className={location.pathname === '/goals' ? 'active' : ''}>Goals</a></li>
                    <li><a href="/stock-analyzer" className={location.pathname === '/stock-analyzer' ? 'active' : ''}>Stock Analyzer</a></li>
                    <li><a href="/settings" className={location.pathname === '/settings' ? 'active' : ''}>Settings</a></li>
                </ul>
            </nav>
            <div className="sidebar-footer">
                <div className="username">Welcome, {fname}</div>
                <button onClick={goToLogin} className="logout-button">Log Out</button>
            </div>
        </div>
    );
}

export default Navbar;