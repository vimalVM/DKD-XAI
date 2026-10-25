import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || '';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

export const isSupabaseConfigured = () => {
  return Boolean(
    supabaseUrl &&
    supabaseAnonKey &&
    supabaseUrl.startsWith('http') &&
    supabaseAnonKey.length > 20
  );
};

export const supabase = isSupabaseConfigured()
  ? createClient(supabaseUrl, supabaseAnonKey)
  : null;

// Built-in Demo accounts for 1-click clinical testing
export const DEMO_USERS = {
  doctor: {
    id: 'demo-dr-jenkins',
    email: 'dr.jenkins@clinic.org',
    user_metadata: {
      full_name: 'Dr. Sarah Jenkins',
      role: 'doctor',
      specialty: 'Nephrology & Diabetes Care'
    }
  },
  patient: {
    id: 'demo-patient-doe',
    email: 'john.doe@patient.net',
    user_metadata: {
      full_name: 'John Doe',
      role: 'patient',
      specialty: 'Individual Patient'
    }
  }
};

const LOCAL_STORAGE_KEY = 'dkd_xai_session';

export const getStoredSession = () => {
  try {
    const raw = localStorage.getItem(LOCAL_STORAGE_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch (e) {
    console.error('Error reading local session', e);
    return null;
  }
};

export const setStoredSession = (user) => {
  if (user) {
    localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(user));
  } else {
    localStorage.removeItem(LOCAL_STORAGE_KEY);
  }
};

export const signUpUser = async ({ email, password, fullName, role }) => {
  if (isSupabaseConfigured() && supabase) {
    const { data, error } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName,
          role: role || 'doctor'
        }
      }
    });
    if (error) throw error;
    if (data?.user) {
      setStoredSession(data.user);
    }
    return data;
  }

  // Local fallback mock when Supabase env keys are pending
  const mockUser = {
    id: 'usr_' + Math.random().toString(36).substring(2, 9),
    email,
    user_metadata: {
      full_name: fullName || (role === 'doctor' ? 'Dr. Clinician' : 'Patient User'),
      role: role || 'doctor'
    }
  };
  setStoredSession(mockUser);
  return { user: mockUser };
};

export const signInUser = async ({ email, password }) => {
  if (isSupabaseConfigured() && supabase) {
    const { data, error } = await supabase.auth.signInWithPassword({
      email,
      password
    });
    if (error) throw error;
    if (data?.user) {
      setStoredSession(data.user);
    }
    return data;
  }

  // Local fallback mock
  const existing = getStoredSession();
  const user = (existing && existing.email === email)
    ? existing
    : {
        id: 'usr_' + Math.random().toString(36).substring(2, 9),
        email,
        user_metadata: {
          full_name: email.includes('dr') ? 'Dr. Clinician' : 'Registered User',
          role: email.includes('dr') ? 'doctor' : 'patient'
        }
      };
  setStoredSession(user);
  return { user };
};

export const signOutUser = async () => {
  if (isSupabaseConfigured() && supabase) {
    await supabase.auth.signOut();
  }
  setStoredSession(null);
};
