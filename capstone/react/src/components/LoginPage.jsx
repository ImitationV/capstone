// LoginPage.jsx
import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../styles/login.css';

function LoginPage() {
    const navigate = useNavigate();

    const goToOverview = () => {
        navigate('/overview');
    };

    return (
        <div className="login-page">
            <div className="login-container">
                <h2>Budget Buddy</h2>
                <div className="login-form">
                    <div className="input-group">
                        <label htmlFor="userId">UserID</label>
                        <input type="text" id="userId" name="userId" />
                    </div>
                    <div className="input-group">
                        <label htmlFor="password">Password</label>
                        <input type="password" id="password" name="password" />
                    </div>
                    <button onClick={goToOverview} className="sign-in-button">Sign In</button>
                    <div className="create-account-link">
                        <a href="#">Create Account</a>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default LoginPage;