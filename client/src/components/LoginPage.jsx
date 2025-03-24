// LoginPage.jsx
import React from 'react';
import { useNavigate} from 'react-router-dom';
import { useState } from 'react';
import '../styles/login.css';
import axios from 'axios';

// Creates an axios instance with a predefined base URL for making API requests
const api = axios.create({
    baseURL: 'http://localhost:4000/api',
});


function LoginPage() {
    const navigate = useNavigate();

    

    const goToOverview = () => {
        navigate('/overview');
    };

    const [username, setUsername] = useState('');
    const [password, setPassword] = useState('');
    const [statusMessage, setStatusMessage] = useState('');

    // Handles the login process
    // It sends a POST request to the '/login' endpoint with the username and password
    // It then checks if the login was successful and updates the status message accordingly
    // If the login fails, it sets the status message to 'Invalid username or password'
    // If an error occurs, it sets the status message to the error message from the response
    const handleLogin = async () => {
        try {

            // Sends a POST request to the '/login' endpoint with the username and password
            // The POST request is sent with:
            // - URL: 'http://localhost:4000/api/login' (baseURL + '/login')
            // - Request body: { userid: username, password: password }
            // - Response format: { success: boolean, message: string }
            //   success indicates if login worked, message contains status/error details
            const response = await api.post('/login', {
                userid: username,
                password: password
            });

            // Checks if the login was successful
            if (response.data.message === 'Login successful') {
                setStatusMessage('Login successful');
                console.log('Login successful');
                goToOverview();
            } else {
                // If the login fails, sets the status message to 'Invalid username or password'
                setStatusMessage('Invalid username or password');
                console.log('Invalid username or password');
            }
        } catch (error) {
            console.log('Error:', error);
            setStatusMessage(error.response?.data?.error || 'An error occurred');
        }
    };
    

    return (
        <div className="login-page">
            <div className="login-container">
                <h2>Budget Buddy</h2>
                <div className="login-form">
                    <div className="input-group">
                        <label htmlFor="userId">UserID</label>
                        {/* This input field captures the username value and updates it using the setUsername state function
                            when the user types. The value is controlled by the username state variable. */}
                        <input type="text" id="userId" name="userId" value={username} onChange={(e) => setUsername(e.target.value)} />
                    </div>
                    <div className="input-group">
                        <label htmlFor="password">Password</label>
                        {/* This input field captures the password value and updates it using the setPassword state function
                            when the user types. The value is controlled by the password state variable. */}
                        <input type="password" id="password" name="password" value={password} onChange={(e) => setPassword(e.target.value)} />
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