import React from 'react';
import Navbar from './Navbar';
import '../styles/layout.css';

import { useLocation } from 'react-router-dom';
import ChatbotPopup from './ChatbotPopup';
import { useLocation } from 'react-router-dom'; 
import Chatbot from './Chatbot';


function Layout({ children }) {
    const location = useLocation(); // gets the current location


    const isLoginPage = location.pathname === '/'; // checks if the current path is the login page ("/")

    return (
        <><div className="layout">
    
    const isLoginPage = location.pathname === '/'; // checks if the current path is the login page ("/")

    return (
        <div className="layout">

            {/* shows the Navbar if the current path is not the login page */}
            {!isLoginPage && <Navbar />}
            <div className="content-area">
                {children}
            </div>

        </div><ChatbotPopup /></>
    );
}

export default Layout;