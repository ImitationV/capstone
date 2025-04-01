import {useState} from 'react';
import '../styles/login.css'; 
import { supabase } from '../supabaseClient';
import {ThemeSupa} from '@supabase/auth-ui-shared';
import {Auth} from '@supabase/auth-ui-react';

{/* Using login.css for styling for now - change later or reflect onto the LoginPage.jsx file */}

function GoogleAuth() {
   const [session, setSession] = useState(null)

   useEffect(() => {
      supabase.auth.getSession().then(({ data: { session } }) => {
        setSession(session)
      })
      const {
        data: { subscription },
      } = supabase.auth.onAuthStateChange((_event, session) => {
        setSession(session)
      })
      return () => subscription.unsubscribe()
    }, [])

    if (!session) {
      return (<Auth supabaseClient={supabase} appearance={{ theme: ThemeSupa }} />)
    }
    else {
      return (<div>Logged in!</div>)
    }

}
