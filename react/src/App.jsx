import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import LoginPage from './components/LoginPage';
import OverviewPage from './components/OverviewPage';
import GoalsPage from './components/GoalsPage';
import TransactionsPage from './components/TransactionsPage';
import StockAnalyzer from './components/StockAnalyzer';
import ProtectedRoute from './components/ProtectedRoute';
import RegisterPage from './components/RegisterPage';
import SettingsPage from './components/SettingsPage';

function App() {
    return (
        <Router>
            <Layout>
                <Routes>
                    <Route path="/" element={<LoginPage />} />
                    <Route 
                        path="/register" 
                        element={
                            <ProtectedRoute>
                                <RegisterPage />
                            </ProtectedRoute>
                        } 
                    />
                    <Route 
                        path="/overview" 
                        element={
                            <ProtectedRoute>
                                <OverviewPage />
                            </ProtectedRoute>
                        } 
                    />
                    <Route 
                        path="/goals" 
                        element={
                            <ProtectedRoute>
                                <GoalsPage />
                            </ProtectedRoute>
                        } 
                    />
                    <Route 
                        path="/transactions" 
                        element={
                            <ProtectedRoute>
                                <TransactionsPage />
                            </ProtectedRoute>
                        } 
                    />
                    <Route 
                        path="/stock-analyzer" 
                        element={
                            <ProtectedRoute>
                                <StockAnalyzer />
                            </ProtectedRoute>
                        } 
                    />
                    <Route 
                        path="/settings" 
                        element={
                            <ProtectedRoute>
                                <SettingsPage />
                            </ProtectedRoute>
                        } 
                    />
                </Routes>
            </Layout>
        </Router>
    );
}

export default App;