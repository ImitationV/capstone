import React from 'react';
import { Formik, Form, Field, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import { createClient } from '@supabase/supabase-js';
import '../styles/goals.css';

// Initialize Supabase client
const supabaseUrl = 'https://idwneflrvwwcwkjlwbkz.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlkd25lZmxydnd3Y3dramx3Ymt6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1ODM1NjcsImV4cCI6MjA1NzE1OTU2N30.RmyMAOfIS1h30ne2E4AT1RB-XWpjA2DN0Bo4FW-9bmQ';
const supabase = createClient(supabaseUrl, supabaseKey);

function GoalsPage() {
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
        const now = new Date().toISOString(); // Gets the current timestamp with timezone
        const currentUserId = 1; // ---------------------------- testID

        try {
            // Inserts data into goals table
            const { error: goalsError } = await supabase
                .from('GOALS')
                .insert([
                    {
                        user_id: currentUserId, 
                        goal_name: values.goalName,
                        target_amount: values.targetAmount,
                        time_frame: values.timeFrame,
                        created_date: now,
                        last_updated: now,
                    },
                ]);

            if (goalsError) {
                console.error('Error inserting into goals table:', goalsError);
                return;
            }

            // Inserts data into the userprofile table
            const { error: userProfileError } = await supabase
                .from('USERPROFILE')
                .upsert([  // Upsert to update or insert
                    {
                        profile_id: currentUserId, // testID
                        income_frequency: values.incomeFrequency,
                        current_savings: values.currentSaving,
                        monthly_expenses: values.monthlyExpense,
                        income: values.income,
                        last_updated: now,
                    },
                ]);

            if (userProfileError) {
                console.error('Error inserting into userprofile table:', userProfileError);
                return;
            }

            console.log('Data submitted successfully!');
            resetForm(); //resets the form after successful submission and displays a success message to the user
        } catch (error) {
            console.error('An unexpected error occurred:', error);
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
                        <h2>Financial Goals & Profile</h2>
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


                        <button type="submit" className="submit-button">Submit</button>
                    </Form>
                )}
            </Formik>
        </div>
    );
}

export default GoalsPage;