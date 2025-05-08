// LoginPage.jsx
import React from 'react';
import { useNavigate} from 'react-router-dom';
import { useState } from 'react';
import '../styles/login.css';
import axios from 'axios';

// Creates an axios instance with a predefined base URL for making API requests
const api = axios.create({
    baseURL: 'http://localhost:4000',
});

function LoginPage() {
    const navigate = useNavigate();
    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    const handleLogin = async () => {
        try {
            console.log('Sending login request with:', { userid: username, password: password });
            const response = await api.post('/api/login', {
                userid: username,
                password: password
            });

            if (response.data.success) {
                setStatusMessage('Login successful');
                console.log('Login successful');
                // Store user information in localStorage
                localStorage.setItem('user', JSON.stringify({
                    id: response.data.user.id,
                    username: response.data.user.username,
                    fname: response.data.user.fname
                }));
                navigate('/overview');
            } else {
                setStatusMessage(response.data.message || 'Invalid username or password');
                console.log('Login failed:', response.data.message);
            }
        } catch (error) {
            console.log('Error:', error);
            setStatusMessage(error.response?.data?.message || 'An error occurred');
        }
    };

    return (
        <div className="login-page">
            <div className="login-container">
                <h2>Budget Buddy</h2>
                <div className="login-form">
                    {statusMessage && (
                        <div className={`status-message ${statusMessage.includes('successful') ? 'success' : 'error'}`}>
                            {statusMessage}
                        </div>
                    )}
                    <div className="input-group">
                        <label htmlFor="username">Username</label>
                        <input 
                            type="text" 
                            id="username" 
                            name="username" 
                            value={username} 
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="Enter your username"
                        />
                    </div>
                    <div className="input-group">
                        <label htmlFor="password">Password</label>
                        <input 
                            type="password" 
                            id="password" 
                            name="password" 
                            value={password} 
                            onChange={(e) => setPassword(e.target.value)}
                            placeholder="Enter your password"
                        />
                    </div>
                    <button type='submit' onClick={handleLogin} className="sign-in-button">Sign In</button>
                    <div className="create-account-link">
                        <a href="#">Create Account</a>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default LoginPage;