import { BrowserRouter as Router, Route, Routes } from 'react-router-dom';
import Layout from './components/Layout';
import LoginPage from './components/LoginPage';
import OverviewPage from './components/OverviewPage';
import GoalsPage from './components/GoalsPage';
import TransactionsPage from './components/TransactionsPage';
import StockAnalyzer from './components/StockAnalyzer';
import RegisterPage from './components/RegisterPage';
import SettingsPage from './components/SettingsPage';

function App() {
    return (
        <Router>
            <Layout>
                <Routes>
                    <Route path="/" element={<LoginPage />} />
                    <Route path="/register" element={<RegisterPage />} />
                    <Route path="/overview" element={<OverviewPage />} />
                    <Route path="/goals" element={<GoalsPage />} />
                    <Route path="/transactions" element={<TransactionsPage />} />
                    <Route path="/stock-analyzer" element={<StockAnalyzer />} />
                    <Route path="/settings" element={<SettingsPage />} />
                </Routes>
            </Layout>
        </Router>
    );
}

export default App;