import {useState, useEffect} from 'react';
import '../styles/login.css'; 
import { supabase } from '../../supabaseClient';
import {ThemeSupa} from '@supabase/auth-ui-shared';
import {Auth} from '@supabase/auth-ui-react';
import { useNavigate } from 'react-router-dom';

{/* Using login.css for styling for now - change later or reflect onto the LoginPage.jsx file */}

function GoogleAuth() {
   const [session, setSession] = useState(null);
   const navigate = useNavigate();

   useEffect(() => {
      // Check for existing session
      supabase.auth.getSession().then(({ data: { session } }) => {
        setSession(session)
        if (session) {
          navigate('/overview');
        }
      });

      // Listen for auth changes
      const {
        data: { subscription },
      } = supabase.auth.onAuthStateChange((_event, session) => {
        setSession(session)
        if (session) {
          navigate('/overview');
        }
      })
      return () => subscription.unsubscribe()
    }, [navigate])

    const handleGoogleSignIn = async () => {
      try {
        const { error } = await supabase.auth.signInWithOAuth({
          provider: "google"
      });
      if (error) throw error;
     } catch (error) {
        console.error('Error signing in with Google:', error);
      }
    };

    if (!session) {
      return (
        <Auth 
          supabaseClient={supabase} 
          appearance={{ theme: ThemeSupa }}
          providers={['google']}
          redirectTo={`${window.location.origin}/overview`}
          view="sign_in"
          showLinks={false}
          onlyThirdPartyProviders={true}
        />
      );
    }

    return null;
}

export default GoogleAuth;