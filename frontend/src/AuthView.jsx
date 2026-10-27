/* eslint-disable react/prop-types */
import { useState } from 'react';
import {
  Activity,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  HeartHandshake,
  LockKeyhole,
  Mail,
  ShieldCheck,
  Sparkles,
  Stethoscope,
  UserRound,
  AlertCircle
} from 'lucide-react';
import {
  signInUser,
  signUpUser,
  setStoredSession,
  DEMO_USERS,
  isSupabaseConfigured
} from './supabaseClient';

const ROLE_DETAILS = {
  doctor: {
    title: 'Doctor',
    portal: 'Clinical workspace',
    description: 'Assess patient vitals and explore explainable risk factors.',
    Icon: Stethoscope,
    firstName: 'Dr. Sarah Jenkins',
    placeholder: 'doctor@hospital.org'
  },
  patient: {
    title: 'Patient',
    portal: 'Personal workspace',
    description: 'Explore your kidney health with clear, supportive guidance.',
    Icon: HeartHandshake,
    firstName: 'John Doe',
    placeholder: 'you@example.com'
  }
};

export default function AuthView({ onAuthSuccess }) {
  const [stage, setStage] = useState('choose');
  const [mode, setMode] = useState('signin');
  const [role, setRole] = useState('doctor');
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  const [successMsg, setSuccessMsg] = useState('');
  const isConfigured = isSupabaseConfigured();
  const roleDetails = ROLE_DETAILS[role];
  const RoleIcon = roleDetails.Icon;

  const openLogin = (nextRole) => {
    setRole(nextRole);
    setStage('login');
    setMode('signin');
    setErrorMsg('');
    setSuccessMsg('');
  };

  const handleDemoLogin = () => {
    const demoUser = DEMO_USERS[role];
    setStoredSession(demoUser);
    onAuthSuccess(demoUser);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setErrorMsg('');
    setSuccessMsg('');
    setLoading(true);

    try {
      if (!email.trim() || !password.trim()) {
        throw new Error('Enter your email address and password to continue.');
      }

      if (mode === 'signup') {
        if (!fullName.trim()) throw new Error('Please enter your full name.');
        if (password.length < 6) throw new Error('Use a password with at least 6 characters.');

        const result = await signUpUser({
          email: email.trim(),
          password,
          fullName: fullName.trim(),
          role
        });

        if (result?.user) {
          setSuccessMsg('Your account is ready. Welcome to DKD-XAI.');
          window.setTimeout(() => onAuthSuccess(result.user), 450);
        }
      } else {
        const result = await signInUser({ email: email.trim(), password, role });
        if (result?.user) {
          setSuccessMsg('Signed in successfully.');
          window.setTimeout(() => onAuthSuccess(result.user), 350);
        }
      }
    } catch (error) {
      console.error('Auth error:', error);
      setErrorMsg(error.message || 'Authentication failed. Please check your details.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className={`entry-page ${stage === 'login' ? 'entry-page-auth' : ''}`}>
      <div className="entry-ambient entry-ambient-mint" />
      <div className="entry-ambient entry-ambient-blue" />
      <div className="entry-grid-pattern" />

      <header className="entry-header">
        <a className="entry-brand" href="#home" aria-label="DKD-XAI home">
          <span className="entry-brand-mark"><img src="/logo.png" alt="" /></span>
          <span className="entry-brand-copy">
            <strong>DKD-XAI</strong>
            <span>Kidney health, made clearer</span>
          </span>
        </a>
        <div className="entry-trust"><ShieldCheck size={15} /> Explainable clinical support</div>
      </header>

      <main className="entry-main" id="home">
        <section className="entry-copy" key={stage + mode + role}>
          {stage === 'choose' ? (
            <div className="entry-content entry-content-choose">
              <div className="entry-eyebrow"><span className="eyebrow-pulse" /> A clearer view of kidney health</div>
              <h1>Care that sees<br /><span>the whole picture.</span></h1>
              <p className="entry-lede">A thoughtful space for diabetic kidney health—bringing clinical insights and everyday guidance together.</p>

              <div className="entry-role-heading">
                <div>
                  <span className="entry-step-label">GET STARTED</span>
                  <h2>Choose your workspace</h2>
                </div>
                <span className="entry-step-count">01 <i /> 02</span>
              </div>

              <div className="entry-role-list">
                {['doctor', 'patient'].map((nextRole) => {
                  const details = ROLE_DETAILS[nextRole];
                  const Icon = details.Icon;
                  return (
                    <button className={`entry-role-card ${nextRole}`} key={nextRole} onClick={() => openLogin(nextRole)}>
                      <span className="entry-role-icon"><Icon size={21} strokeWidth={1.8} /></span>
                      <span className="entry-role-info">
                        <strong>{details.title} login</strong>
                        <span>{details.description}</span>
                      </span>
                      <span className="entry-role-arrow"><ArrowRight size={17} /></span>
                    </button>
                  );
                })}
              </div>
              <div className="entry-note"><LockKeyhole size={14} /> Your workspace is private and secure.</div>
            </div>
          ) : (
            <div className="entry-content entry-content-login">
              <button className="entry-back-button" type="button" onClick={() => { setStage('choose'); setErrorMsg(''); setSuccessMsg(''); }}>
                <ArrowLeft size={15} /> Back to workspaces
              </button>

              <div className={`entry-login-heading ${role}`}>
                <span className="entry-login-icon"><RoleIcon size={20} /></span>
                <div><span>{roleDetails.portal}</span><h1>{mode === 'signin' ? `Welcome back, ${roleDetails.title.toLowerCase()}.` : 'Create your account.'}</h1></div>
              </div>
              <p className="entry-login-lede">{mode === 'signin' ? 'Sign in to continue to your kidney health workspace.' : `Join DKD-XAI as a ${roleDetails.title.toLowerCase()} and get started.`}</p>

              <div className="entry-auth-tabs" role="tablist" aria-label="Account access">
                <button type="button" role="tab" aria-selected={mode === 'signin'} className={mode === 'signin' ? 'active' : ''} onClick={() => { setMode('signin'); setErrorMsg(''); setSuccessMsg(''); }}>Sign in</button>
                <button type="button" role="tab" aria-selected={mode === 'signup'} className={mode === 'signup' ? 'active' : ''} onClick={() => { setMode('signup'); setErrorMsg(''); setSuccessMsg(''); }}>Create account</button>
              </div>

              {errorMsg && <div className="entry-alert error" role="alert"><AlertCircle size={16} /><span>{errorMsg}</span></div>}
              {successMsg && <div className="entry-alert success" role="status"><CheckCircle2 size={16} /><span>{successMsg}</span></div>}

              <form className="entry-form" onSubmit={handleSubmit}>
                {mode === 'signup' && <label className="entry-field"><span>Full name</span><div className="entry-input"><UserRound size={17} /><input type="text" autoComplete="name" placeholder={roleDetails.firstName} value={fullName} onChange={(event) => setFullName(event.target.value)} required /></div></label>}
                <label className="entry-field"><span>Email address</span><div className="entry-input"><Mail size={17} /><input type="email" autoComplete="email" placeholder={roleDetails.placeholder} value={email} onChange={(event) => setEmail(event.target.value)} required /></div></label>
                <label className="entry-field"><span>Password</span><div className="entry-input"><LockKeyhole size={17} /><input type="password" autoComplete={mode === 'signin' ? 'current-password' : 'new-password'} placeholder="At least 6 characters" value={password} onChange={(event) => setPassword(event.target.value)} minLength={6} required /></div></label>
                <button type="submit" disabled={loading} className={`entry-submit ${role}`}>
                  {loading ? <><span className="entry-spinner" /> {mode === 'signup' ? 'Creating account…' : 'Signing in…'}</> : <>{mode === 'signup' ? 'Create account' : `Sign in as ${roleDetails.title.toLowerCase()}`} <ArrowRight size={17} /></>}
                </button>
              </form>

              <button type="button" className="entry-demo-button" onClick={handleDemoLogin}><Sparkles size={15} /> Try the {roleDetails.title.toLowerCase()} demo <ArrowRight size={14} /></button>
              {!isConfigured && <p className="entry-demo-hint"><Activity size={13} /> Demo access is available while account services are being configured.</p>}
            </div>
          )}
        </section>

        <section className="entry-visual" aria-label="Kidney health illustration">
          <div className="entry-orbit entry-orbit-one" />
          <div className="entry-orbit entry-orbit-two" />
          <div className="entry-visual-glow" />
          <div className="entry-image-wrap"><img className="entry-kidney-image" src="/kidney-hero.png" alt="Illustration of healthy kidneys with a pulse line" /></div>
          <div className="entry-floating-card entry-floating-top"><span className="floating-icon mint"><Activity size={16} /></span><span><strong>Whole-person view</strong><small>Vitals · risk · guidance</small></span><span className="floating-live-dot" /></div>
          <div className="entry-floating-card entry-floating-bottom"><span className="floating-icon blue"><Sparkles size={16} /></span><span><strong>AI, made explainable</strong><small>Every insight has context</small></span><ArrowRight size={15} className="floating-card-arrow" /></div>
          <div className="entry-visual-caption"><span>01</span><i /> <span>CARE, CONNECTED</span></div>
          <div className="entry-visual-stamp"><span>DKD</span><small>Kidney care<br />intelligence</small></div>
        </section>
      </main>

      <footer className="entry-footer">
        <span>DKD-XAI <i>•</i> Explainable kidney health support</span>
        <span><ShieldCheck size={13} /> For informed conversations with your care team</span>
      </footer>
    </div>
  );
}
