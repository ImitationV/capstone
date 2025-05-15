import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import '../styles/login.css';

const api = axios.create({
    baseURL: 'http://localhost:4000',
});

function RegisterPage() {
    const navigate = useNavigate();
    const [form, setForm] = useState({
        email: '',
        fname: '',
        lname: '',
        username: '',
        password: ''
    });
    const [statusMessage, setStatusMessage] = useState('');
    const [loading, setLoading] = useState(false);

    const handleChange = (e) => {
        setForm({ ...form, [e.target.name]: e.target.value });
    };

    const handleRegister = async (e) => {
        e.preventDefault();
        setStatusMessage('');
        setLoading(true);
        try {
            // Registration API call
            const response = await api.post('/api/register', form);
            if (response.data.success) {
                setStatusMessage('Registration successful! Logging you in...');
                // Auto-login after registration
                const loginResp = await api.post('/api/login', {
                    userid: form.username,
                    password: form.password
                });
                if (loginResp.data.success) {
                    localStorage.setItem('user', JSON.stringify({
                        id: loginResp.data.user.id,
                        username: loginResp.data.user.username,
                        fname: loginResp.data.user.fname
                    }));
                    navigate('/overview');
                } else {
                    setStatusMessage('Registration succeeded but login failed. Please try logging in.');
                }
            } else {
                setStatusMessage(response.data.message || 'Registration failed.');
            }
        } catch (error) {
            setStatusMessage(error.response?.data?.message || 'An error occurred during registration.');
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="login-page">
            <div className="login-container">
                <h2>Create Account</h2>
                <form className="login-form" onSubmit={handleRegister}>
                    {statusMessage && (
                        <div className={`status-message ${statusMessage.includes('successful') ? 'success' : 'error'}`}>
                            {statusMessage}
                        </div>
                    )}
                    <div className="input-group">
                        <label htmlFor="email">Email</label>
                        <input type="email" id="email" name="email" value={form.email} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                        <label htmlFor="fname">First Name</label>
                        <input type="text" id="fname" name="fname" value={form.fname} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                        <label htmlFor="lname">Last Name</label>
                        <input type="text" id="lname" name="lname" value={form.lname} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                        <label htmlFor="username">Username</label>
                        <input type="text" id="username" name="username" value={form.username} onChange={handleChange} required />
                    </div>
                    <div className="input-group">
                        <label htmlFor="password">Password</label>
                        <input type="password" id="password" name="password" value={form.password} onChange={handleChange} required />
                    </div>
                    <button type="submit" className="sign-in-button" disabled={loading}>
                        {loading ? 'Registering...' : 'Create Account'}
                    </button>
                </form>
                <button
                    type="button"
                    className="sign-in-button"
                    style={{ marginTop: '1rem', background: '#ccc', color: '#222' }}
                    onClick={() => navigate('/')}
                >
                    Back to Login
                </button>
            </div>
        </div>
    );
}

export default RegisterPage; 