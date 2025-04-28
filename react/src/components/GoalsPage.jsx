import React from 'react';
import { Formik, Form, Field, ErrorMessage } from 'formik';
import * as Yup from 'yup';
import '../styles/goals.css';

function GoalsPage() {
    const initialValues = {
        monthlyIncome: '',
        savings: '',
        monthlyExpenses: '',
        monthlyLoanPayment: '',
    };

    const validationSchema = Yup.object({
        monthlyIncome: Yup.number().required('Monthly Income is required').positive('Must be a positive number'),
        savings: Yup.number().required('Savings is required').positive('Must be a positive number'),
        monthlyExpenses: Yup.number().required('Monthly Expenses is required').positive('Must be a positive number'),
        monthlyLoanPayment: Yup.number().required('Monthly Loan Payment is required').positive('Must be a positive number'),
    });

    const onSubmit = (values) => {
        console.log('Form values submitted:', values);
        // display on the browser console the values submitted
    };

    return (
        <div className="goals-page-container">
            <h2>Spending Goal</h2>
            <Formik
                initialValues={initialValues}
                validationSchema={validationSchema}
                onSubmit={onSubmit}
            >
                <Form className="goals-form">
                    <div className="form-group">
                        <label htmlFor="monthlyIncome">Monthly Income</label>
                        <Field type="number" id="monthlyIncome" name="monthlyIncome" />
                        <ErrorMessage name="monthlyIncome" component="div" className="error-message" />
                    </div>

                    <div className="form-group">
                        <label htmlFor="savings">Savings</label>
                        <Field type="number" id="savings" name="savings" />
                        <ErrorMessage name="savings" component="div" className="error-message" />
                    </div>

                    <div className="form-group">
                        <label htmlFor="monthlyExpenses">Monthly Expenses</label>
                        <Field type="number" id="monthlyExpenses" name="monthlyExpenses" />
                        <ErrorMessage name="monthlyExpenses" component="div" className="error-message" />
                    </div>

                    <div className="form-group">
                        <label htmlFor="monthlyLoanPayment">Monthly Loan Payment</label>
                        <Field type="number" id="monthlyLoanPayment" name="monthlyLoanPayment" />
                        <ErrorMessage name="monthlyLoanPayment" component="div" className="error-message" />
                    </div>

                    <button type="submit" className="submit-button">Submit</button>
                </Form>
            </Formik>
        </div>
    );
}

export default GoalsPage;