import React, { useEffect, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom'; 
import { supabase } from '../../supabaseClient';
import '../styles/navbar.css';

function Navbar() {
    const navigate = useNavigate();
    const location = useLocation();
    const [fname, setFname] = useState('');
    const [isGoogleUser, setIsGoogleUser] = useState(false);

    useEffect(() => {
        // Check if user is logged in via Google Auth
        supabase.auth.getSession().then(({ data: { session } }) => {
            setIsGoogleUser(!!session);
            if (session?.user?.user_metadata?.full_name) {
                setFname(session.user.user_metadata.full_name);
            }
        });

        // Get user information from localStorage (for regular login)
        const user = JSON.parse(localStorage.getItem('user'));
        if (user) {
            setFname(user.fname);
        }
    }, []);

    const handleLogout = async () => {
        try {
            if (isGoogleUser) {
                // Handle Google Auth logout
                const { error } = await supabase.auth.signOut();
                if (error) throw error;
            }
            
            // Always clear localStorage
            localStorage.removeItem('user');
            navigate('/');
        } catch (error) {
            console.error('Error during logout:', error.message);
        }
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
                <button onClick={handleLogout} className="logout-button">Log Out</button>
            </div>
        </div>
    );
}

export default Navbar;