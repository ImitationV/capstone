import React from 'react';
import Navbar from './Navbar';
import '../styles/layout.css';
import { useLocation } from 'react-router-dom';
import ChatbotPopup from './ChatbotPopup';


function Layout({ children }) {
    const location = useLocation(); // gets the current location

    // Check if current path is login page or GoogleAuth page
    const isLoginOrAuthPage = location.pathname === '/' || location.pathname === '/googleauth';

    return (
        <div className="layout">

            {/* shows the Navbar if the current path is not a login or auth page */}
            {!isLoginOrAuthPage && <Navbar />}
            <div className="content-area">
                {children}
            </div>

            {/* Only show ChatbotPopup if NOT on login or auth page */}
            {!isLoginOrAuthPage && <ChatbotPopup />}
        </div>
    );
}

export default Layout;