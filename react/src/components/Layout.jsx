import React, { useState, useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import Navbar from './Navbar';
import '../styles/layout.css';
import { supabase } from '../../supabaseClient';
import ChatbotPopup from './ChatbotPopup';

function Layout({ children }) {
    const location = useLocation();
    const [isAuthenticated, setIsAuthenticated] = useState(false);

    useEffect(() => {
        // Check authentication status
        const checkAuth = async () => {
            const { data: { session } } = await supabase.auth.getSession();
            const localUser = JSON.parse(localStorage.getItem('user'));
            console.log('Layout - Current session:', session); // Debug session
            console.log('Layout - Local user:', localUser); // Debug local storage
            console.log('Layout - Current path:', location.pathname); // Debug navigation
            setIsAuthenticated(!!session || !!localUser);
        };

        checkAuth();
    }, [location]);

    // Don't show navbar on login page
    if (location.pathname === '/') {
        return <div className="content-area">{children}</div>;
    }

    return (
        <div className="layout">
            {isAuthenticated && <Navbar />}
            <div className="content-area">
                {children}
            </div>

            {/* Only show ChatbotPopup if NOT on login or auth page */}
            {!isAuthenticated && <ChatbotPopup />}
        </div>
    );
}

export default Layout;