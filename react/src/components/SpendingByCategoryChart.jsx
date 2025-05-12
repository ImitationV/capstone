import React, { useState, useEffect, useCallback } from 'react';
import { PieChart, Pie, Cell, Tooltip, Legend, ResponsiveContainer, Sector } from 'recharts';
import { supabase } from './supabaseClient';
import '../styles/SpendingByCategoryChart.css';

// color palette
const COLORS = ['#4CAF50', '#2196F3', '#FF9800', '#F44336', '#9C27B0', '#795548', '#00BCD4', '#FFEB3B'];

// Custom active shape for hover effect
const renderActiveShape = (props) => {
  const RADIAN = Math.PI / 180;
  const { cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent, value } = props;
  const sin = Math.sin(-RADIAN * midAngle);
  const cos = Math.cos(RADIAN * midAngle);
  const sx = cx + (outerRadius + 10) * cos;
  const sy = cy + (outerRadius + 10) * sin;
  const mx = cx + (outerRadius + 30) * cos;
  const my = cy + (outerRadius + 30) * sin;
  const ex = mx + (cos >= 0 ? 1 : -1) * 22;
  const ey = my;
  const textAnchor = cos >= 0 ? 'start' : 'end';

  return (
    <g>
      <text x={cx} y={cy} dy={8} textAnchor="middle" fill={fill} style={{ fontWeight: 'bold' }}>
        {payload.name}
      </text>
      <Sector
        cx={cx}
        cy={cy}
        innerRadius={innerRadius}
        outerRadius={outerRadius + 5}
        startAngle={startAngle}
        endAngle={endAngle}
        fill={fill}
      />
      <Sector
        cx={cx}
        cy={cy}
        startAngle={startAngle}
        endAngle={endAngle}
        innerRadius={outerRadius + 12}
        outerRadius={outerRadius + 14}
        fill={fill}
      />
       <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
       <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
       <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="#333">{`$${value.toFixed(2)}`}</text>
       <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} dy={18} textAnchor={textAnchor} fill="#999">
         {`(${(percent * 100).toFixed(2)}%)`}
       </text>
    </g>
  );
};


const SpendingByCategoryChart = ({ userId }) => {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeIndex, setActiveIndex] = useState(-1); 

  const onPieEnter = useCallback((_, index) => {
    setActiveIndex(index);
  }, [setActiveIndex]);

  const onPieLeave = useCallback(() => {
    setActiveIndex(-1);
  }, [setActiveIndex]);


  useEffect(() => {
    const fetchSpendingData = async () => {
      if (!userId) {
        setLoading(false);
        setError("User ID is not available.");
        return;
      }
      try {
        const { data: transactions, error } = await supabase
          .from('TRANSACTIONS')
          .select('amount, category')
          .eq('user_id', userId)
          .eq('transaction_type', 'expense'); 

        if (error) throw error;

        if (!transactions || transactions.length === 0) {
            setData([]); // No data to display
            setLoading(false);
            return;
        }

        // Aggregate spending by category
        const spendingByCategory = transactions.reduce((acc, transaction) => {
          const category = transaction.category || 'Uncategorized';
          const amount = Math.abs(parseFloat(transaction.amount)); // Ensure amount is positive and a number
          if (!isNaN(amount)) {
             acc[category] = (acc[category] || 0) + amount;
          }
          return acc;
        }, {});

        const formattedData = Object.keys(spendingByCategory).map(category => ({
          name: category,
          value: spendingByCategory[category],
        }));

        formattedData.sort((a, b) => b.value - a.value);

        setData(formattedData);
      } catch (err) {
          console.error("Error fetching spending data:", err);
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchSpendingData();

  }, [userId]); // Refetch if userId changes

  if (loading) return <p>Loading spending data...</p>;
  if (error) return <p>Error loading spending data: {error}</p>;


  return (
    <div className="chart-container"> 
      <h2 className="chart-title">Spending by Category</h2> 
      {data.length === 0 ? (
         <p>No spending data available for this user.</p>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              activeIndex={activeIndex}
              activeShape={renderActiveShape} // Custom shape for the highlighted slice
              data={data}
              cx="50%"
              cy="50%"
              innerRadius={60} // donut shape
              outerRadius={90}
              fill="#8884d8" 
              dataKey="value"
              isAnimationActive={true}
              onMouseEnter={onPieEnter}
              onMouseLeave={onPieLeave}
            >
              {data.map((entry, index) => (
                <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
              ))}
            </Pie>
            <Tooltip
               formatter={(value, name) => [`$${parseFloat(value).toFixed(2)}`, name]} // Ensure value is formatted as currency
            />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      )}
    </div>
  );
};

export default SpendingByCategoryChart;