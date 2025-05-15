import { createClient } from '@supabase/supabase-js';
const supabaseUrl = 'https://idwneflrvwwcwkjlwbkz.supabase.co';
const supabaseKey = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imlkd25lZmxydnd3Y3dramx3Ymt6Iiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDE1ODM1NjcsImV4cCI6MjA1NzE1OTU2N30.RmyMAOfIS1h30ne2E4AT1RB-XWpjA2DN0Bo4FW-9bmQ';
export const supabase = createClient(supabaseUrl, supabaseKey); // Exported as 'supabase'