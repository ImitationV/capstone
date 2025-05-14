import React from 'react';
import Navbar from './Navbar';
import '../styles/layout.css';
import { useLocation } from 'react-router-dom';
import ChatbotPopup from './ChatbotPopup';


function Layout({ children }) {
    const location = useLocation(); // gets the current location


    const isLoginOrRegisterPage = location.pathname === '/' || location.pathname === '/register';

    // Only show ChatbotPopup if user is signed in
    const isSignedIn = Boolean(localStorage.getItem('user'));

    return (
        <div className="layout">

            {/* shows the Navbar if the current path is not the login or register page */}
            {!isLoginOrRegisterPage && <Navbar />}
            <div className="content-area">
                {children}
            </div>

            {isSignedIn && <ChatbotPopup />}
        </div>
    );
}

export default Layout;