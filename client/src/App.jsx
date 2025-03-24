import React from 'react';
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import LoginPage from './components/LoginPage';
import OverviewPage from './components/OverviewPage';
import GoalsPage from './components/GoalsPage';
import TransactionsPage from './components/TransactionsPage';
import axios from 'axios';
import { useEffect } from 'react';


// Creates an axios instance with a predefined base URL for making API requests
const api = axios.create({
    baseURL: 'http://localhost:4000/api',
});



function App() {
    // fetchApi makes an HTTP GET request to the backend health check endpoint
    // using the configured axios instance. It verifies the connection between
    // the frontend and backend by logging the response data to the console.
    const fetchApi = async () => {
        const response = await api.get('/health');
        console.log(response.data);
    }
 
    // useEffect hook is used to perform side effects in function components
    // In this case, it calls fetchApi() when the component mounts
    // The empty dependency array [] means it only runs once on mount
    // Without the dependency array, it would run on every render
    // With dependencies in the array, it would run whenever those values change
    useEffect(() => {
        fetchApi();
    }, []);


    return (
        <Router>
            <Layout>
                <Routes>
                    <Route path="/" element={<LoginPage />} />
                    <Route path="/overview" element={<OverviewPage />} />
                    <Route path="/goals" element={<GoalsPage />} />
                    <Route path="/transactions" element={<TransactionsPage />} />
                </Routes>
            </Layout>
        </Router>
    );
}

export default App;