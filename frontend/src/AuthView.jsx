import React, { useState } from 'react';
import {
  Stethoscope,
  HeartHandshake,
  ShieldCheck,
  Mail,
  Lock,
  User,
  ArrowRight,
  Sparkles,
  CheckCircle2,
  AlertCircle,
  Database
} from 'lucide-react';
import {
  signInUser,
  signUpUser,
  setStoredSession,
  DEMO_USERS,
  isSupabaseConfigured
} from './supabaseClient';

export default function AuthView({ onAuthSuccess }) {
  const [mode, setMode] = useState('signin'); // 'signin' | 'signup'
  const [role, setRole] = useState('doctor'); // 'doctor' | 'patient'
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');

  const isConfigured = isSupabaseConfigured();

  const handleDemoLogin = (demoRole) => {
    const demoUser = DEMO_USERS[demoRole];
    setStoredSession(demoUser);
    onAuthSuccess(demoUser);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setErrorMsg('');
    setSuccessMsg('');
    setLoading(true);

    try {
      if (mode === 'signup') {
        if (!fullName.trim()) {
          throw new Error('Please enter your full name.');
        }
        if (!email.trim() || !password.trim()) {
          throw new Error('Please enter email and password.');
        }
        if (password.length < 6) {
          throw new Error('Password should be at least 6 characters.');
        }

        const res = await signUpUser({
          email: email.trim(),
          password,
          fullName: fullName.trim(),
          role
        });

        if (res?.user) {
          setSuccessMsg('Account created successfully! Welcome aboard.');
          setTimeout(() => {
            onAuthSuccess(res.user);
          }, 600);
        }
      } else {
        if (!email.trim() || !password.trim()) {
          throw new Error('Please enter email and password.');
        }

        const res = await signInUser({
          email: email.trim(),
          password
        });

        if (res?.user) {
          setSuccessMsg('Signed in successfully.');
          setTimeout(() => {
            onAuthSuccess(res.user);
          }, 400);
        }
      }
    } catch (err) {
      console.error('Auth error:', err);
      setErrorMsg(err.message || 'Authentication failed. Please verify credentials.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-page-container">
      {/* Background ambient lighting */}
      <div className="auth-ambient-glow-1" />
      <div className="auth-ambient-glow-2" />

      <div className="auth-card-wrapper">
        {/* Top Header & Branding */}
        <div className="auth-brand-header">
          <div className="auth-logo-badge">
            <img src="/logo.png" alt="DKD-XAI" className="auth-logo-img" />
          </div>
          <h1 className="auth-title">DKD-XAI</h1>
          <p className="auth-subtitle">
            Explainable AI for Diabetic Kidney Disease Risk
          </p>
        </div>

        {/* Role Selection Cards */}
        <div className="auth-role-section">
          <label className="auth-role-label">Choose your access portal</label>
          <div className="auth-role-cards-grid">
            <button
              type="button"
              className={`auth-role-card ${role === 'doctor' ? 'active-doctor' : ''}`}
              onClick={() => setRole('doctor')}
            >
              <div className="auth-role-icon-box doctor-icon-box">
                <Stethoscope size={22} />
              </div>
              <div className="auth-role-info">
                <div className="auth-role-header-row">
                  <span className="auth-role-name">Doctor</span>
                  <span className="auth-role-pill doctor-pill">Clinical Portal</span>
                </div>
                <p className="auth-role-desc">
                  Detailed SHAP risk attribution, clinical biomarkers, ADA 2026/KDIGO surveillance plans.
                </p>
              </div>
              {role === 'doctor' && <CheckCircle2 className="auth-role-check" size={18} />}
            </button>

            <button
              type="button"
              className={`auth-role-card ${role === 'patient' ? 'active-patient' : ''}`}
              onClick={() => setRole('patient')}
            >
              <div className="auth-role-icon-box patient-icon-box">
                <HeartHandshake size={22} />
              </div>
              <div className="auth-role-info">
                <div className="auth-role-header-row">
                  <span className="auth-role-name">Normal User / Patient</span>
                  <span className="auth-role-pill patient-pill">Layman Portal</span>
                </div>
                <p className="auth-role-desc">
                  Simple, compassionate, plain-language kidney checks with zero confusing medical jargon.
                </p>
              </div>
              {role === 'patient' && <CheckCircle2 className="auth-role-check" size={18} />}
            </button>
          </div>
        </div>

        {/* Auth Mode Tabs (Sign In / Register) */}
        <div className="auth-tabs-slider">
          <button
            type="button"
            className={`auth-tab-btn ${mode === 'signin' ? 'active-tab' : ''}`}
            onClick={() => {
              setMode('signin');
              setErrorMsg('');
              setSuccessMsg('');
            }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`auth-tab-btn ${mode === 'signup' ? 'active-tab' : ''}`}
            onClick={() => {
              setMode('signup');
              setErrorMsg('');
              setSuccessMsg('');
            }}
          >
            Create Account
          </button>
        </div>

        {/* Feedback messages */}
        {errorMsg && (
          <div className="auth-alert error-alert">
            <AlertCircle size={16} />
            <span>{errorMsg}</span>
          </div>
        )}
        {successMsg && (
          <div className="auth-alert success-alert">
            <CheckCircle2 size={16} />
            <span>{successMsg}</span>
          </div>
        )}

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="auth-form">
          {mode === 'signup' && (
            <div className="auth-input-group">
              <label htmlFor="auth-fullname">Full Name</label>
              <div className="auth-input-wrapper">
                <User size={18} className="auth-input-icon" />
                <input
                  id="auth-fullname"
                  type="text"
                  placeholder={role === 'doctor' ? 'e.g. Dr. Robert Vance' : 'e.g. Eleanor Davis'}
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  required
                />
              </div>
            </div>
          )}

          <div className="auth-input-group">
            <label htmlFor="auth-email">Email Address</label>
            <div className="auth-input-wrapper">
              <Mail size={18} className="auth-input-icon" />
              <input
                id="auth-email"
                type="email"
                placeholder={role === 'doctor' ? 'clinician@hospital.org' : 'patient@gmail.com'}
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </div>

          <div className="auth-input-group">
            <label htmlFor="auth-password">Password</label>
            <div className="auth-input-wrapper">
              <Lock size={18} className="auth-input-icon" />
              <input
                id="auth-password"
                type="password"
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className={`auth-submit-btn ${role === 'doctor' ? 'btn-doctor' : 'btn-patient'}`}
          >
            {loading ? (
              <span className="auth-spinner">Processing...</span>
            ) : (
              <>
                <span>
                  {mode === 'signup'
                    ? `Register as ${role === 'doctor' ? 'Doctor' : 'Patient'}`
                    : `Sign In as ${role === 'doctor' ? 'Doctor' : 'Patient'}`}
                </span>
                <ArrowRight size={18} />
              </>
            )}
          </button>
        </form>

        {/* Quick Demo 1-Click Access */}
        <div className="auth-demo-divider">
          <span>Or test immediately with 1-click preset</span>
        </div>

        <div className="auth-demo-buttons-row">
          <button
            type="button"
            className="auth-demo-btn demo-doc-btn"
            onClick={() => handleDemoLogin('doctor')}
          >
            <Stethoscope size={15} />
            <span>⚡ Demo Doctor (Dr. Jenkins)</span>
          </button>
          <button
            type="button"
            className="auth-demo-btn demo-pat-btn"
            onClick={() => handleDemoLogin('patient')}
          >
            <HeartHandshake size={15} />
            <span>⚡ Demo Patient (John Doe)</span>
          </button>
        </div>

        {/* Footer info & Supabase status badge */}
        <div className="auth-footer-badge">
          <Database size={13} />
          <span>
            {isConfigured
              ? 'Supabase Cloud Auth Connected'
              : 'Supabase Ready (Provide VITE_SUPABASE_URL in frontend/.env)'}
          </span>
        </div>
      </div>
    </div>
  );
}
