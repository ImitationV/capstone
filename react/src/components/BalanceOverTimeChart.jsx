import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { supabase } from './supabaseClient';
import '../styles/BalanceOverTimeChart.css';

// function to get the date string (YYYY-MM-DD) from a timestamp
const getDateString = (timestamp) => {
  const date = new Date(timestamp);
  return date.getUTCFullYear() + '-' +
         ('0' + (date.getUTCMonth() + 1)).slice(-2) + '-' +
         ('0' + date.getUTCDate()).slice(-2);
};

// function to format timestamp to readable date string for labels
const formatTimestampToDateLabel = (timestamp) => {
  const date = new Date(timestamp);
  if (!isNaN(date.getTime())) {
     return date.toLocaleDateString();
  }
  return '';
};

const BalanceOverTimeChart = ({ userId , onBalanceUpdate}) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchAndCalculateBalanceData = async () => {
      if (!userId) {
        setLoading(false);
        setError("User ID is not available.");
        return;
      }

      try {
        // Fetch the initial savings from USERPROFILE
        const { data: userProfile, error: profileError } = await supabase
          .from('USERPROFILE')
          .select('current_savings')
          .eq('profile_id', userId)
          .single(); // Assuming one profile per user

        if (profileError && profileError.code !== 'PGRST116') { // PGRST116 means no row found
            console.warn("No user profile found or error fetching profile:", profileError);
        }

        // Initialize current balance with savings, default to 0 if no profile or savings
        let currentBalance = userProfile ? parseFloat(userProfile.current_savings) || 0 : 0;
        if (isNaN(currentBalance)) currentBalance = 0; // Ensure it's a number


        // Fetch transactions for the user, ordered by date
        const { data: transactions, error: transactionsError } = await supabase
          .from('TRANSACTIONS')
          .select('amount, date_created, transaction_type')
          .eq('user_id', userId) 
          .order('date_created', { ascending: true });

        if (transactionsError) throw transactionsError;

        // Calculate running balance points after each transaction
        const intermediateData = [];
        let runningBalance = currentBalance; 

        if (transactions && transactions.length > 0) {
            transactions.forEach(transaction => {
                const amount = parseFloat(transaction.amount);
                const timestamp = new Date(transaction.date_created).getTime();

                if (!isNaN(amount) && !isNaN(timestamp)) {
                   if (transaction.transaction_type === 'income') {
                       runningBalance += amount;
                   } else if (transaction.transaction_type === 'expense') {
                       runningBalance -= Math.abs(amount); // Ensure subtraction for expense
                   }

                   intermediateData.push({
                       date: timestamp, // Use timestamp
                       balance: runningBalance,
                       transactionDate: transaction.date_created
                   });
                } else {
                    console.warn("Skipping transaction with invalid amount or date:", transaction);
                }
            });
        }

        // Aggregate data points to keep only the last balance for each day
        const aggregatedDataMap = {}; 

        const firstTransactionTimestamp = intermediateData.length > 0 ? intermediateData[0].date : null;
        const startDateObj = firstTransactionTimestamp ? new Date(firstTransactionTimestamp) : new Date();
        // Set to the start of the day before the first transaction day, or start of today/fixed date
        startDateObj.setUTCHours(0, 0, 0, 0);
        if (firstTransactionTimestamp) {
             startDateObj.setUTCDate(startDateObj.getUTCDate() - 1);
        } else {
             // If no transactions, just use a fixed reasonable past date
             startDateObj.setUTCFullYear(startDateObj.getUTCFullYear() - 1);
        }
        const startDateTimestamp = startDateObj.getTime();


         aggregatedDataMap[getDateString(startDateTimestamp)] = {
             date: startDateTimestamp,
             dateLabel: 'Start',
             balance: currentBalance // initial balance
         };


        intermediateData.forEach(point => {
            const dayString = getDateString(point.date);
            aggregatedDataMap[dayString] = {
                 date: point.date, 
                 dateLabel: formatTimestampToDateLabel(point.date),
                 balance: point.balance // This is the running balance after the last transaction of the day
            };
        });

        // Converts the map values back to an array and sort by timestamp
        const finalChartData = Object.values(aggregatedDataMap).sort((a, b) => a.date - b.date);

        setData(finalChartData);
        
        //Call the callback with the last balance if it exists
        if (finalChartData.length > 0){
          const lastBalance = finalChartData[finalChartData.length - 1].balance;
          onBalanceUpdate(lastBalance); // Call the callback with the last balance
        }
      } catch (err) {
        console.error("Error fetching or calculating balance data:", err);
        setError(err.message);
        setData([]); 
      } finally {
        setLoading(false);
      }
    };

    fetchAndCalculateBalanceData();

  }, [userId]); // Refetch if userId changes

  // Data Label Formatter for Tooltip
  // This function formats the tooltip data when hovering over a point
  const formatTooltip = (value, name, props) => {
      const point = props.payload;
      if (point && typeof point.balance !== 'undefined') {
          const tooltipTitle = point.dateLabel || 'Balance';
          return [`$${parseFloat(point.balance).toFixed(2)}`, tooltipTitle]; // Format value, use dateLabel as name
      }
      return [value, name]; 
  };

  // Custom XAxis tick formatter 
  const formatXAxisTick = (tickItem) => {
      if (typeof tickItem === 'number') {
          // Check if it's the special 'Start' timestamp
           if (data.length > 0 && tickItem === data[0].date && data[0].dateLabel === 'Start') {
              return 'Start';
          }
           // Format other timestamps
          const date = new Date(tickItem);
           if (!isNaN(date.getTime())) {
             return date.toLocaleDateString();
           }
      }
      return ''; // Return empty string for invalid ticks
  };


  if (loading) return <p>Loading balance data...</p>;
  if (error) return <p>Error loading balance data: {error}</p>;

   const hasMeaningfulData = data.length > 1 || (data.length === 1 && data[0]?.balance !== 0);

  if (!hasMeaningfulData) {
       return <p>No balance history or initial savings available for this user.</p>;
  }

  return (
    <div className="chart-container"> 
      <h2 className="chart-title">Balance Over Time</h2> 
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" vertical={false} /> 
          {/* Use 'date' (timestamp) as dataKey for XAxis */}
          <XAxis
              dataKey="date"
              tickFormatter={formatXAxisTick}
              type="number" 
              scale="time" 
              domain={['dataMin', 'dataMax']}
              minTickGap={20} 
              padding={{ left: 20, right: 20 }} 
              allowDuplicatedCategory={false} 
          />
          <YAxis tickFormatter={(value) => `$${parseFloat(value).toFixed(0)}`} /> 
          <Tooltip formatter={formatTooltip} labelFormatter={formatXAxisTick} />
          <Legend />
          <Line
            type= "monotone"
            dataKey="balance"
            stroke="#3b82f6" 
            strokeWidth={2} 
            dot={true} 
            activeDot={{ // Style dots on hover
              r: 6, 
              stroke: '#3b82f6', 
              strokeWidth: 2, 
              fill: '#fff', 
              cursor: 'pointer' // Show pointer cursor on hover
            }}
            isAnimationActive={true} 
            animationDuration={800} // Animation speed
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
};

export default BalanceOverTimeChart;