const express = require('express');
const router = express.Router();

require('dotenv').config();

const userid = process.env.USER_ID;
const password = process.env.PASSWORD;
const user = {
    userid: userid,
    password: password
}


router.post('/api/login', (req, res) => {
    // Destructures userid and password from the request body (req.body)
    // This extracts the values sent from the frontend login form
    const { userid, password } = req.body;

    console.log('userid entered: ',userid, '\tpassword entered: ',password);

    // Compares the entered values with the stored user credentials
    if (userid === user.userid && password === user.password) {
        console.log('Login successful');
        res.json({ success: true, message: 'Login successful' });
    } else {
        console.log('Login failed');
        res.json({ success: false, message: 'Invalid username or password' });
    };

})


module.exports = router;