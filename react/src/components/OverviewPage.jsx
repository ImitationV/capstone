import React from 'react';
import '../styles/overview.css';

import ChatbotPopup from './ChatbotPopup';
function OverviewPage() {

    return (
        <><div className="overview-page-content">


            <div className="header">
                <h1>Overview</h1>
            </div>
            <div className="balance-section">
                <div className="balance-box">
                    <h3>Balance</h3>
                    <p>$101.69</p>
                </div>
                <div className="savings-box">
                    <h3>savings-box</h3>
                    <p>$50000.00</p>
                </div>
                <div className="menu-box">
                    <h3>menu-box</h3>
                    <ul>
                        <p>Income</p>
                    </ul>
                </div>
                <div className="menu-box">
                    <h3>menu-box</h3>
                    <ul>
                        <p>Income</p>
                    </ul>
                </div>
            </div>
            <div className="charts-section">
                <div className="chart-box">
                    <h3>Balance over Time</h3>
                    <p>chart-box</p>
                </div>
                <div className="chart-box">
                    <h3>Spending in Categories</h3>
                    <p>chart-box</p>
                    <a href="#" className="view-all-link">View all</a>
                </div>
            </div>
        </div><ChatbotPopup /></>
    );
}

export default OverviewPage;