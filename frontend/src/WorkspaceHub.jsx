/* eslint-disable react/prop-types */
import { useState } from 'react';
import {
  Activity,
  ArrowRight,
  ArrowUpRight,
  Bot,
  Heart,
  LogOut,
  ShieldCheck,
  Sparkles,
  Stethoscope
} from 'lucide-react';

const OPTIONS = [
  {
    id: 'clinical',
    title: 'Clinical Vital Forms',
    description: 'Enter health details and explore an explainable risk assessment.',
    Icon: Activity,
    tone: 'clinical',
    label: '01 / ASSESS'
  },
  {
    id: 'ai',
    title: 'AI Assist',
    description: 'Ask kidney health questions in a thoughtful, guided conversation.',
    Icon: Bot,
    tone: 'ai',
    label: '02 / EXPLORE'
  }
];

export default function WorkspaceHub({ currentUser, onOpen, onSignOut }) {
  const [launching, setLaunching] = useState(null);
  const name = currentUser?.user_metadata?.full_name || 'there';
  const role = currentUser?.user_metadata?.role === 'patient' ? 'patient' : 'doctor';
  const RoleIcon = role === 'doctor' ? Stethoscope : Heart;

  const openOption = (option) => {
    if (launching) return;
    setLaunching(option.id);
    const delay = window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 0 : 610;
    window.setTimeout(() => onOpen(option.id), delay);
  };

  const handleSignOut = () => {
    onSignOut();
  };

  return (
    <div className={`hub-page ${launching ? 'hub-launching' : ''} ${launching ? `launch-${launching}` : ''}`}>
      <div className="hub-ambient hub-ambient-a" />
      <div className="hub-ambient hub-ambient-b" />
      <header className="hub-header">
        <div className="hub-brand">
          <span className="hub-brand-logo"><img src="/logo.png" alt="" /></span>
          <span><strong>DKD-XAI</strong><small>Kidney health workspace</small></span>
        </div>
        <div className="hub-header-right">
          <div className="hub-session-pill"><span className="hub-session-dot" /> Secure session</div>
          <div className="hub-user-chip">
            <span className={`hub-user-icon ${role}`}><RoleIcon size={15} /></span>
            <span className="hub-user-copy"><strong>{name}</strong><small>{role === 'doctor' ? 'Doctor' : 'Patient'}</small></span>
            <button type="button" className="hub-signout" title="Sign out" aria-label="Sign out" onClick={handleSignOut}><LogOut size={16} /></button>
          </div>
        </div>
      </header>

      <main className="hub-main">
        <section className="hub-intro">
          <div className="hub-eyebrow"><span /> YOUR KIDNEY HEALTH SPACE</div>
          <h1>A clearer picture<br />starts <em>here.</em></h1>
          <p className="hub-lede">Choose how you’d like to continue. Your clinical tools and AI support are ready when you are.</p>
          <div className="hub-welcome-card">
            <span className="hub-welcome-avatar">{name.slice(0, 1).toUpperCase()}</span>
            <span><small>WELCOME BACK</small><strong>{name}</strong></span>
            <span className="hub-welcome-check"><ShieldCheck size={17} /></span>
          </div>
          <div className="hub-care-note"><span className="hub-care-icon"><Heart size={14} fill="currentColor" /></span><span>Designed to support care decisions<br />with clear, evidence-grounded insights.</span></div>
        </section>

        <section className="hub-stage" aria-label="Choose a kidney health tool">
          <div className="hub-stage-topline"><span>YOUR NEXT STEP</span><span className="hub-stage-mark">DKD-XAI <i /> 02 TOOLS</span></div>
          <div className="hub-art-scene">
            <div className="hub-art-halo" />
            <div className="hub-art-ring hub-art-ring-one" />
            <div className="hub-art-ring hub-art-ring-two" />
            <img className="hub-art-image" src="/kidney-hero.png" alt="Illustration of healthy kidneys" />
            <div className="hub-image-caption"><span className="caption-pulse" /> HEALTH, IN BALANCE</div>
            <div className="hub-floating-stat stat-left"><span className="stat-icon"><Activity size={15} /></span><span><small>Whole picture</small><strong>Vitals + context</strong></span></div>
            <div className="hub-floating-stat stat-right"><span className="stat-icon sparkle"><Sparkles size={15} /></span><span><small>Thoughtful AI</small><strong>Clear explanations</strong></span></div>
            <div className="hub-pulse-line"><span /></div>
          </div>
          <div className="hub-option-list">
            {OPTIONS.map(({ id, title, description, Icon, tone, label }) => (
              <article
                key={id}
                className={`hub-option ${tone} ${launching === id ? 'selected' : ''}`}
              >
                <div className="hub-option-head">
                  <span className="hub-option-icon"><Icon size={19} strokeWidth={1.9} /></span>
                  <span className="hub-option-text"><small>{label}</small><strong>{title}</strong><span>{description}</span></span>
                  <span className="hub-option-go" aria-hidden="true"><ArrowUpRight size={17} /></span>
                </div>
                <button
                  type="button"
                  className={`hub-option-cta ${tone}`}
                  onClick={() => openOption({ id })}
                  disabled={Boolean(launching)}
                  aria-label={`Open ${title}`}
                >
                  {id === 'clinical' ? 'Open clinical forms' : 'Start AI Assist'}
                  <ArrowRight size={15} />
                </button>
              </article>
            ))}
          </div>
          <div className="hub-stage-footer"><span><ShieldCheck size={14} /> Private, secure workspace</span><span>Select a tool to continue <ArrowRight size={13} /></span></div>
        </section>
      </main>

      <footer className="hub-footer"><span>DKD-XAI <i>•</i> Explainable kidney health support</span><span>For informed conversations with your care team</span></footer>
      {launching && <div className="hub-transition-wash" aria-hidden="true"><span className="hub-transition-mark"><img src="/logo.png" alt="" /></span></div>}
    </div>
  );
}
