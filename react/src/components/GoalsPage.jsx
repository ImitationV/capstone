import React, { useEffect, useState } from 'react';
import { Formik, Form, Field, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import { supabase } from './supabaseClient';
import '../styles/goals.css';


function GoalsPage() {
    const [userGoals, setUserGoals] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    // Get the current user's ID from localStorage
    const getCurrentUserId = () => {
        const user = JSON.parse(localStorage.getItem('user'));
        return user ? user.id : null;
    };

    // Delete a goal
    const deleteGoal = async (goalId) => {
        try {
            const { error } = await supabase
                .from('GOALS')
                .delete()
                .eq('goal_id', goalId);

            if (error) throw error;
            
            // Update the goals list after deletion
            setUserGoals(userGoals.filter(goal => goal.goal_id !== goalId));
        } catch (err) {
            console.error('Error deleting goal:', err);
            setError('Failed to delete goal');
        }
    };

    // Fetch user's goals
    const fetchUserGoals = async () => {
        const userId = getCurrentUserId();
        if (!userId) {
            setError('Please log in to view your goals');
            setLoading(false);
            return;
        }

        try {
            const { data, error } = await supabase
                .from('GOALS')
                .select('*')
                .eq('user_id', userId)
                .order('created_date', { ascending: false });

            if (error) throw error;
            setUserGoals(data);
        } catch (err) {
            console.error('Error fetching goals:', err);
            setError('Failed to fetch goals');
        } finally {
            setLoading(false);
        }
    };

    // Fetch goals when component mounts
    useEffect(() => {
        fetchUserGoals();
    }, []);

    const initialValues = {
        goalName: '',
        targetAmount: '',
        timeFrame: '',
        incomeFrequency: '',
        currentSaving: '',
        monthlyExpense: '',
        income: '',
    };

    const validationSchema = Yup.object({
        goalName: Yup.string().required('Goal Name is required'),
        targetAmount: Yup.number().required('Target Amount is required').positive('Must be a positive number'),
        timeFrame: Yup.date().required('Time Frame is required'),
        incomeFrequency: Yup.string().oneOf(['Weekly', 'Biweekly', 'Monthly'], 'Invalid frequency').required('Income Frequency is required'),
        currentSaving: Yup.number().required('Current Saving is required').positive('Must be a positive number'),
        monthlyExpense: Yup.number().required('Monthly Expense is required').positive('Must be a positive number'),
        income: Yup.number().required('Income is required').positive('Must be a positive number'),
    });

    const onSubmit = async (values, { resetForm }) => {
        const now = new Date().toISOString();
        const userId = getCurrentUserId();

        if (!userId) {
            setError('Please log in to create goals');
            return;
        }

        try {
            // Insert new goal
            const { error: goalsError } = await supabase
                .from('GOALS')
                .insert([
                    {
                        user_id: userId,
                        goal_name: values.goalName,
                        target_amount: parseInt(values.targetAmount),
                        time_frame: new Date(values.timeFrame).toISOString(),
                        created_date: now,
                        last_updated: now,
                    },
                ]);

            if (goalsError) {
                console.error('Error saving goal:', goalsError);
                setError('Failed to save goal: ' + goalsError.message);
                return;
            }

            // Update user profile
            const { error: userProfileError } = await supabase
                .from('USERPROFILE')
                .upsert([
                    {
                        old_profile_id: userId,
                        income_frequency: values.incomeFrequency,
                        current_savings: parseFloat(values.currentSaving),
                        monthly_expenses: parseFloat(values.monthlyExpense),
                        income: parseFloat(values.income),
                        last_updated: now,
                    },
                ]);

            if (userProfileError) {
                console.error('Error updating profile:', userProfileError);
                // Don't set error here since the goal was saved successfully
            }

            // Refresh goals list
            await fetchUserGoals();
            resetForm();
        } catch (error) {
            console.error('Error in onSubmit:', error);
            setError('An unexpected error occurred');
        }
    };

    return (
        <div className="goals-page-container">
            <Formik
                initialValues={initialValues}
                validationSchema={validationSchema}
                onSubmit={onSubmit}
            >
                {({ errors, touched }) => (
                    <Form className="goals-form">
                        <h2>Create New Goal</h2>
                        <div className="form-group">
                            <label htmlFor="goalName">Goal Name</label>
                            <Field type="text" id="goalName" name="goalName" />
                            <ErrorMessage name="goalName" component="div" className="error-message" />
                        </div>

                        <div className="form-group">
                            <label htmlFor="income">Income</label>
                            <Field type="number" id="income" name="income" />
                            <ErrorMessage name="income" component="div" className="error-message" />
                        </div>

                        <div className="form-group">
                            <label htmlFor="incomeFrequency">Income Frequency</label>
                            <Field as="select" id="incomeFrequency" name="incomeFrequency">
                                <option value="">Select Frequency</option>
                                <option value="Weekly">Weekly</option>
                                <option value="Biweekly">Biweekly</option>
                                <option value="Monthly">Monthly</option>
                            </Field>
                            <ErrorMessage name="incomeFrequency" component="div" className="error-message" />
                        </div>

                        <div className="form-group">
                            <label htmlFor="currentSaving">Current Saving</label>
                            <Field type="number" id="currentSaving" name="currentSaving" />
                            <ErrorMessage name="currentSaving" component="div" className="error-message" />
                        </div>

                        <div className="form-group">
                            <label htmlFor="monthlyExpense">Monthly Expense</label>
                            <Field type="number" id="monthlyExpense" name="monthlyExpense" />
                            <ErrorMessage name="monthlyExpense" component="div" className="error-message" />
                        </div>

                        <div className="form-group">
                            <label htmlFor="targetAmount">Target Amount</label>
                            <Field type="number" id="targetAmount" name="targetAmount" />
                            <ErrorMessage name="targetAmount" component="div" className="error-message" />
                        </div>

                        <div className="form-group">
                            <label htmlFor="timeFrame">Time Frame</label>
                            <Field type="date" id="timeFrame" name="timeFrame" />
                            <ErrorMessage name="timeFrame" component="div" className="error-message" />
                        </div>

                        <button type="submit" className="submit-button">Create Goal</button>
                    </Form>
                )}
            </Formik>

            <div className="goals-list">
                <h2>Your Financial Goals</h2>
                {loading ? (
                    <p>Loading your goals...</p>
                ) : error ? (
                    <p className="error-message">{error}</p>
                ) : userGoals.length === 0 ? (
                    <p>No goals set yet. Create your first goal above!</p>
                ) : (
                    <div className="goals-grid">
                        {userGoals.map((goal) => (
                            <div key={goal.goal_id} className="goal-card">
                                <div className="goal-header">
                                    <h3>{goal.goal_name}</h3>
                                    <button 
                                        onClick={() => deleteGoal(goal.goal_id)}
                                        className="delete-button"
                                        title="Delete goal"
                                    >
                                        ×
                                    </button>
                                </div>
                                <p>Target: ${goal.target_amount.toLocaleString()}</p>
                                <p>Deadline: {new Date(goal.time_frame).toLocaleDateString()}</p>
                                <p>Created: {new Date(goal.created_date).toLocaleDateString()}</p>
                            </div>
                        ))}
                    </div>
                )}
            </div>
        </div>
    );
}

export default GoalsPage;