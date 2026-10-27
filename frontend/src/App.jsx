import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  Activity, 
  Send, 
  ShieldCheck, 
  User, 
  Bot, 
  ExternalLink, 
  AlertTriangle, 
  CheckCircle2, 
  Stethoscope, 
  Sparkles,
  RotateCcw,
  SlidersHorizontal,
  LogOut,
  HeartHandshake,
  ArrowLeft
} from 'lucide-react';
import AuthView from './AuthView';
import WorkspaceHub from './WorkspaceHub';
import { getStoredSession, signOutUser } from './supabaseClient';

const API_BASE = (import.meta.env.VITE_API_URL || '').replace(/\/$/, '');

const DEFAULT_SAMPLES = [
  {
    id: "sample-1",
    name: "High Risk Case (71yo Male, HbA1c 9.5%, SBP 158, On Insulin)",
    description: "71yo male with 16 years of diabetes, HbA1c 9.5%, BP 158/94, BMI 34, on insulin.",
    data: {
      age: 71, female: 0, hba1c: 9.5, bmi: 34.0, waist_cm: 108,
      sbp: 158, dbp: 94, hdl: 32, total_chol: 220,
      dm_diagnosed: 1, insulin_use: 1, diabetes_pills: 0, dm_duration_years: 16
    }
  },
  {
    id: "sample-2",
    name: "Low Risk Case (38yo Female, HbA1c 6.3%, SBP 115, Diet Controlled)",
    description: "38yo female with 2 years of diabetes, HbA1c 6.3%, BP 115/75, BMI 23.5, no insulin.",
    data: {
      age: 38, female: 1, hba1c: 6.3, bmi: 23.5, waist_cm: 78,
      sbp: 115, dbp: 75, hdl: 58, total_chol: 165,
      dm_diagnosed: 1, insulin_use: 0, diabetes_pills: 0, dm_duration_years: 2
    }
  },
  {
    id: "sample-3",
    name: "Borderline Case (52yo Female, HbA1c 8.9%, SBP 126, Metformin)",
    description: "52yo female with 6 years of diabetes, HbA1c 8.9%, BP 126/80, on metformin pills, no insulin.",
    data: {
      age: 52, female: 1, hba1c: 8.9, bmi: 29.2, waist_cm: 94,
      sbp: 126, dbp: 80, hdl: 44, total_chol: 195,
      dm_diagnosed: 1, insulin_use: 0, diabetes_pills: 1, dm_duration_years: 6
    }
  },
  {
    id: "sample-4",
    name: "Complex Case (68yo Female, 14y Diabetes, SBP 134, On Insulin)",
    description: "68yo female with 14-yr diabetes duration on insulin, SBP 134 mmHg, BMI 38.9.",
    data: {
      age: 68, female: 1, hba1c: 5.7, bmi: 38.9, waist_cm: 110,
      sbp: 134, dbp: 82, hdl: 29, total_chol: 193,
      dm_diagnosed: 1, insulin_use: 1, diabetes_pills: 0, dm_duration_years: 14
    }
  }
];

const EMPTY_PATIENT = {
  age: null,
  female: null,
  hba1c: null,
  bmi: null,
  waist_cm: null,
  sbp: null,
  dbp: null,
  hdl: null,
  total_chol: null,
  dm_diagnosed: null,
  insulin_use: null,
  diabetes_pills: null,
  dm_duration_years: null
};

export default function App() {
  const [currentUser, setCurrentUser] = useState(() => getStoredSession());
  const [activeWorkspace, setActiveWorkspace] = useState(null);
  const userRole = currentUser?.user_metadata?.role || 'doctor';
  const userName = currentUser?.user_metadata?.full_name || (userRole === 'doctor' ? 'Dr. Sarah Jenkins' : 'John Doe');

  const getInitialGreeting = (r) => {
    if (r === 'patient') {
      return (
        "Hello! I am your **Kidney Health Assistant**.\n\n" +
        "I'm here to help you understand your kidney health and how diabetes affects your body, in simple, friendly, everyday words.\n\n" +
        "You can share your recent numbers (e.g. *'I am 58 years old, diabetes for 8 years, HbA1c 7.9%, blood pressure 135/85'*), " +
        "pick an example case from the dropdown above, or ask any question about keeping your kidneys healthy!"
      );
    }
    return (
      "Hello! I am your **DKD-XAI Clinical AI Assistant**.\n\n" +
      "You can describe a patient naturally in clinical terms, for example:\n" +
      "> *\"71yo male with 16 years of diabetes, HbA1c 9.5%, BP 158/94, BMI 34, on insulin. What is his kidney risk?\"*\n\n" +
      "Or pick any patient preset from the dropdown above to view an instant risk assessment."
    );
  };

  const [task, setTask] = useState('dkd_any');
  const [inputMode, setInputMode] = useState('form'); // 'form' | 'chat'
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: getInitialGreeting(userRole),
      prediction: null
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentPatient, setCurrentPatient] = useState(EMPTY_PATIENT);
  const [prediction, setPrediction] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState('');
  const [predictionAnimationKey, setPredictionAnimationKey] = useState(0);
  const predictionRequestIdRef = useRef(0);
  const [samples, setSamples] = useState(DEFAULT_SAMPLES);
  const chatHistoryRef = useRef(null);
  const isAutoScrollPinnedRef = useRef(true);

  // Detect when user manually scrolls up in chat
  const handleChatScroll = () => {
    if (!chatHistoryRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = chatHistoryRef.current;
    const distanceFromBottom = scrollHeight - (scrollTop + clientHeight);
    isAutoScrollPinnedRef.current = distanceFromBottom <= 80;
  };

  const runDirectPrediction = useCallback(async (patientData, activeTask, currentRole = userRole) => {
    // Only run if essential clinical markers are present:
    const hasCore = patientData && patientData.age != null && (patientData.hba1c != null || patientData.dm_diagnosed === 1);
    if (!hasCore) {
      setPrediction(null);
      setPredictionLoading(false);
      setPredictionError('');
      return null;
    }

    const requestId = ++predictionRequestIdRef.current;
    setPredictionLoading(true);
    setPredictionError('');
    setPrediction(null);

    try {
      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ patient: patientData, task: activeTask, role: currentRole })
      });
      const data = await response.json().catch(() => null);

      if (!response.ok) {
        const errorMsg = data?.detail?.message || `The prediction service returned an error (${response.status}).`;
        throw new Error(errorMsg);
      }
      if (!data || !Number.isFinite(Number(data.risk_percentage))) {
        throw new Error('The prediction service returned an incomplete result. Please try again.');
      }
      if (requestId !== predictionRequestIdRef.current) return null;

      setPrediction(data);
      setPredictionAnimationKey((key) => key + 1);
      return data;
    } catch (error) {
      if (requestId !== predictionRequestIdRef.current) return null;
      console.error('Prediction failed:', error);
      setPrediction(null);
      setPredictionError(
        error instanceof TypeError
          ? 'Could not connect to the prediction service. Check the API connection and try again.'
          : (error.message || 'Unable to calculate risk. Please try again.')
      );
      return null;
    } finally {
      if (requestId === predictionRequestIdRef.current) setPredictionLoading(false);
    }
  }, [userRole]);

  // Fetch example patients
  useEffect(() => {
    fetch(`${API_BASE}/api/sample-patients`)
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(err => console.error("Could not load samples", err));
  }, []);

  // Keep the risk estimate in sync with the active profile and assessment.
  useEffect(() => {
    runDirectPrediction(currentPatient, task, userRole);
  }, [currentPatient, task, userRole, runDirectPrediction]);

  // Smart auto-scroll: only pins to bottom if user has not scrolled up
  useEffect(() => {
    if (isAutoScrollPinnedRef.current && chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
    }
  }, [messages, loading]);



  const streamingTimerRef = useRef(null);

  const stopStreaming = () => {
    if (streamingTimerRef.current) {
      clearInterval(streamingTimerRef.current);
      streamingTimerRef.current = null;
    }
  };

  useEffect(() => {
    return () => stopStreaming();
  }, []);

  const handleResetChat = (r = userRole) => {
    stopStreaming();
    isAutoScrollPinnedRef.current = true;
    setCurrentPatient(EMPTY_PATIENT);
    setPrediction(null);
    setPredictionError('');
    setMessages([
      {
        role: 'assistant',
        content: getInitialGreeting(r),
        prediction: null
      }
    ]);
    if (chatHistoryRef.current) {
      chatHistoryRef.current.scrollTop = 0;
    }
  };

  const handleSendMessage = async (textToSend, patientOverride) => {
    const text = textToSend || inputMessage;
    if (!text.trim() || loading) return;

    stopStreaming();
    isAutoScrollPinnedRef.current = true;
    const patientToSend = patientOverride || currentPatient;
    const userMsg = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setLoading(true);

    setTimeout(() => {
      if (chatHistoryRef.current) {
        chatHistoryRef.current.scrollTop = chatHistoryRef.current.scrollHeight;
      }
    }, 10);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          task: task,
          current_patient: patientToSend,
          role: userRole
        })
      });

      const data = await res.json();
      if (data.patient) {
        setCurrentPatient(data.patient);
      }
      if (data.prediction) {
        predictionRequestIdRef.current += 1;
        setPrediction(data.prediction);
        setPredictionError('');
        setPredictionLoading(false);
        setPredictionAnimationKey((key) => key + 1);
      } else {
        setPrediction(null);
        setPredictionLoading(false);
        setPredictionError('');
      }

      setLoading(false);

      // Start realistic streaming typewriter animation
      const fullReply = data.reply || "";
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: '',
          prediction: data.prediction,
          isStreaming: true
        }
      ]);

      let charIdx = 0;
      const step = 4;
      streamingTimerRef.current = setInterval(() => {
        charIdx += step;
        if (charIdx >= fullReply.length) {
          stopStreaming();
          setMessages(prev => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last && last.role === 'assistant') {
              copy[copy.length - 1] = {
                ...last,
                content: fullReply,
                isStreaming: false
              };
            }
            return copy;
          });
        } else {
          const slice = fullReply.slice(0, charIdx);
          setMessages(prev => {
            const copy = [...prev];
            const last = copy[copy.length - 1];
            if (last && last.role === 'assistant') {
              copy[copy.length - 1] = {
                ...last,
                content: slice,
                isStreaming: true
              };
            }
            return copy;
          });
        }
      }, 16);

    } catch (err) {
      console.error(err);
      setLoading(false);
      setMessages(prev => [
        ...prev,
        {
          role: 'assistant',
          content: "Sorry, I could not connect to the backend server. Please verify the API is running.",
          prediction: null
        }
      ]);
    }
  };

  const handleSelectSample = (sampleId) => {
    const found = samples.find(s => s.id === sampleId);
    if (!found) return;
    setCurrentPatient(found.data);
    handleSendMessage(`Analyze benchmark patient: ${found.name} (${found.description})`, found.data);
  };

  const handleVitalChange = (field, val) => {
    const updated = { ...currentPatient, [field]: val === '' ? null : Number(val) };
    setCurrentPatient(updated);
  };

  // Helper function to format bold, italic, and inline elements
  const renderFormattedText = (text) => {
    if (!text) return null;
    const parts = text.split(/(\*\*.*?\*\*|\*[^*\n]+?\*)/g);
    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return <strong key={i}>{part.slice(2, -2)}</strong>;
      }
      if (part.startsWith('*') && part.endsWith('*') && part.length > 2) {
        return <em key={i} className="chat-em">{part.slice(1, -1)}</em>;
      }
      return part;
    });
  };

  // Helper function to render structured markdown messages
  const renderMessageContent = (content) => {
    if (!content) return null;
    const lines = content.split('\n');
    const elements = [];
    let currentList = [];

    const flushList = () => {
      if (currentList.length > 0) {
        elements.push(
          <ul key={`ul-${elements.length}`} className="chat-bullet-list">
            {currentList.map((item, idx) => (
              <li key={idx} className="chat-list-item">{renderFormattedText(item)}</li>
            ))}
          </ul>
        );
        currentList = [];
      }
    };

    lines.forEach((line, idx) => {
      const trimmed = line.trim();
      if (!trimmed) {
        flushList();
        return;
      }

      if (trimmed.startsWith('### ')) {
        flushList();
        elements.push(
          <h3 key={`h3-${idx}`} className="chat-h3">
            {renderFormattedText(trimmed.replace('### ', ''))}
          </h3>
        );
      } else if (trimmed.startsWith('#### ')) {
        flushList();
        elements.push(
          <h4 key={`h4-${idx}`} className="chat-h4">
            {renderFormattedText(trimmed.replace('#### ', ''))}
          </h4>
        );
      } else if (trimmed.startsWith('> ')) {
        flushList();
        elements.push(
          <blockquote key={`quote-${idx}`} className="chat-quote">
            {renderFormattedText(trimmed.replace('> ', ''))}
          </blockquote>
        );
      } else if (trimmed.startsWith('*( ') || (trimmed.startsWith('*(') && trimmed.endsWith(')*'))) {
        flushList();
        elements.push(
          <div key={`foot-${idx}`} className="chat-footnote">
            {trimmed.replace(/^\*\(/, '').replace(/\)\*$/, '')}
          </div>
        );
      } else if (trimmed.startsWith('- ') || trimmed.startsWith('• ') || trimmed.startsWith('* ')) {
        const itemText = trimmed.replace(/^[-•*]\s*/, '');
        currentList.push(itemText);
      } else {
        flushList();
        elements.push(
          <p key={`p-${idx}`} className="chat-paragraph">
            {renderFormattedText(trimmed)}
          </p>
        );
      }
    });

    flushList();
    return elements;
  };

  // Clean risk metrics
  const riskVal = prediction ? prediction.risk_percentage : 0;
  const isHighRisk = prediction ? prediction.high_risk : false;
  const thresholdVal = prediction ? prediction.threshold_percentage : 51.3;

  if (!currentUser) {
    return (
      <AuthView
        onAuthSuccess={(user) => {
          setCurrentUser(user);
          setActiveWorkspace(null);
          const r = user?.user_metadata?.role || 'doctor';
          handleResetChat(r);
          runDirectPrediction(currentPatient, task, r);
        }}
      />
    );
  }

  if (!activeWorkspace) {
    return (
      <WorkspaceHub
        currentUser={currentUser}
        onOpen={(workspace) => {
          setInputMode(workspace === 'ai' ? 'chat' : 'form');
          setActiveWorkspace(workspace);
        }}
        onSignOut={async () => {
          await signOutUser();
          setCurrentUser(null);
          setActiveWorkspace(null);
        }}
      />
    );
  }

  return (
    <div className="app-container app-workspace-enter">
      {/* Top Navigation Bar */}
      <header className="app-header">
        <div className="brand-wrapper">
          <img src="/logo.png" alt="DKD-XAI Logo" className="brand-logo-img" />
          <div>
            <div className="brand-title">
              DKD-XAI{' '}
              <span className={`brand-badge ${userRole}`}>
                {userRole === 'doctor' ? 'Clinical Assistant' : 'Patient Assistant'}
              </span>
            </div>
            <div className="brand-subtitle">
              {userRole === 'doctor'
                ? 'Explainable AI for Early Diabetic Kidney Disease Risk'
                : 'Simple, Friendly Kidney Health Guide & Screening'}
            </div>
          </div>
        </div>

        {/* Clean Controls in Header */}
        <div className="header-controls">
          <button
            type="button"
            className="header-workspace-button"
            onClick={() => setActiveWorkspace(null)}
            aria-label="Return to workspace choices"
          >
            <ArrowLeft size={15} /> <span>Workspace</span>
          </button>
          {/* Preset Patients Dropdown */}
          <div className="preset-dropdown-wrap">
            <select 
              className="preset-select"
              onChange={(e) => {
                if (e.target.value) handleSelectSample(e.target.value);
              }}
              defaultValue=""
            >
              <option value="" disabled>Load Example Patient...</option>
              {samples.map(s => (
                <option key={s.id} value={s.id}>{s.name}</option>
              ))}
            </select>
          </div>

          {/* Task Toggle */}
          <div className="task-selector">
            <button 
              className={`task-btn ${task === 'dkd_any' ? 'active' : ''}`}
              onClick={() => setTask('dkd_any')}
            >
              Any Damage
            </button>
            <button 
              className={`task-btn ${task === 'early_dkd' ? 'active' : ''}`}
              onClick={() => setTask('early_dkd')}
            >
              Early DKD vs None
            </button>
          </div>
        </div>

        {/* User Profile Pill & Sign Out in Right Corner */}
        <div className="header-user-corner">
          <div className="user-profile-pill">
            <div className={`user-role-badge ${userRole}`}>
              {userRole === 'doctor' ? <Stethoscope size={13} /> : <HeartHandshake size={13} />}
              <span>{userRole === 'doctor' ? 'Doctor' : 'Patient'}</span>
            </div>
            <span className="user-name-label" title={userName}>
              {userName}
            </span>
            <button 
              type="button" 
              className="signout-btn" 
              title="Sign Out"
              onClick={async () => {
                await signOutUser();
                setCurrentUser(null);
              }}
            >
              <LogOut size={14} />
            </button>
          </div>
        </div>
      </header>

      {/* Main 2-Column Clean Workspace */}
      <div className="main-layout-streamlined">
        
        {/* Left Column: Form & Chat Workspace */}
        <main className="panel-card main-workspace-panel">
          <div className="panel-header">
            {/* Mode Switcher Tabs */}
            <div className="mode-toggle-group">
              <button 
                className={`mode-toggle-btn ${inputMode === 'form' ? 'active' : ''}`}
                onClick={() => setInputMode('form')}
              >
                <SlidersHorizontal size={14} /> Clinical Vital Forms
              </button>
              <button 
                className={`mode-toggle-btn ${inputMode === 'chat' ? 'active' : ''}`}
                onClick={() => setInputMode('chat')}
              >
                <Bot size={14} /> AI Assist
              </button>
            </div>

            {inputMode === 'chat' && (
              <button 
                className="btn-text-action"
                onClick={handleResetChat}
                title="Start a new patient consultation"
              >
                <RotateCcw size={13} /> Clear Chat
              </button>
            )}
          </div>

          {inputMode === 'form' ? (
            <div className="vitals-form-view">
              <div className="vitals-form-scroll">
                
                {/* Form Group 1: Demographics & History */}
                <div className="form-card-section">
                  <div className="form-section-header">
                    <User size={15} color="#0F766E" />
                    <span>Demographics & Diabetes History</span>
                  </div>
                  <div className="form-inputs-grid">
                    <div className="vital-field-box">
                      <label>Age (years)</label>
                      <input 
                        type="number" 
                        value={currentPatient.age ?? ''} 
                        onChange={e => handleVitalChange('age', e.target.value)} 
                        placeholder="e.g. 68"
                      />
                    </div>
                    <div className="vital-field-box">
                      <label>Biological Sex</label>
                      <select 
                        value={currentPatient.female ?? 1} 
                        onChange={e => handleVitalChange('female', e.target.value)}
                      >
                        <option value={1}>Female</option>
                        <option value={0}>Male</option>
                      </select>
                    </div>
                    <div className="vital-field-box">
                      <label>Diabetes Duration (yrs)</label>
                      <input 
                        type="number" 
                        value={currentPatient.dm_duration_years ?? ''} 
                        onChange={e => handleVitalChange('dm_duration_years', e.target.value)} 
                        placeholder="e.g. 14"
                      />
                    </div>
                    <div className="vital-field-box">
                      <label>Diagnosed Diabetes</label>
                      <select 
                        value={currentPatient.dm_diagnosed ?? 1} 
                        onChange={e => handleVitalChange('dm_diagnosed', e.target.value)}
                      >
                        <option value={1}>Yes (Diagnosed)</option>
                        <option value={0}>No (Undiagnosed)</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Form Group 2: Glycemic Control & Medications */}
                <div className="form-card-section">
                  <div className="form-section-header">
                    <Activity size={15} color="#2563EB" />
                    <span>Glycemic Control & Pharmacotherapy</span>
                  </div>
                  <div className="form-inputs-grid">
                    <div className="vital-field-box">
                      <label>HbA1c (%)</label>
                      <input 
                        type="number" 
                        step="0.1"
                        value={currentPatient.hba1c ?? ''} 
                        onChange={e => handleVitalChange('hba1c', e.target.value)} 
                        placeholder="e.g. 8.5"
                      />
                    </div>
                    <div className="vital-field-box">
                      <label>Insulin Use</label>
                      <select 
                        value={currentPatient.insulin_use ?? 0} 
                        onChange={e => handleVitalChange('insulin_use', e.target.value)}
                      >
                        <option value={1}>Yes (On Insulin)</option>
                        <option value={0}>No (No Insulin)</option>
                      </select>
                    </div>
                    <div className="vital-field-box">
                      <label>Oral Diabetes Pills</label>
                      <select 
                        value={currentPatient.diabetes_pills ?? 0} 
                        onChange={e => handleVitalChange('diabetes_pills', e.target.value)}
                      >
                        <option value={1}>Yes (Metformin, etc.)</option>
                        <option value={0}>No</option>
                      </select>
                    </div>
                  </div>
                </div>

                {/* Form Group 3: Blood Pressure & Anthropometry */}
                <div className="form-card-section">
                  <div className="form-section-header">
                    <Stethoscope size={15} color="#0F766E" />
                    <span>Blood Pressure & Anthropometry</span>
                  </div>
                  <div className="form-inputs-grid">
                    <div className="vital-field-box">
                      <label>Systolic BP (mmHg)</label>
                      <input 
                        type="number" 
                        value={currentPatient.sbp ?? ''} 
                        onChange={e => handleVitalChange('sbp', e.target.value)} 
                        placeholder="e.g. 138"
                      />
                    </div>
                    <div className="vital-field-box">
                      <label>Diastolic BP (mmHg)</label>
                      <input 
                        type="number" 
                        value={currentPatient.dbp ?? ''} 
                        onChange={e => handleVitalChange('dbp', e.target.value)} 
                        placeholder="e.g. 84"
                      />
                    </div>
                    <div className="vital-field-box">
                      <label>BMI (kg/m²)</label>
                      <input 
                        type="number" 
                        step="0.1"
                        value={currentPatient.bmi ?? ''} 
                        onChange={e => handleVitalChange('bmi', e.target.value)} 
                        placeholder="e.g. 32.4"
                      />
                    </div>
                    <div className="vital-field-box">
                      <label>Waist (cm)</label>
                      <input 
                        type="number" 
                        value={currentPatient.waist_cm ?? ''} 
                        onChange={e => handleVitalChange('waist_cm', e.target.value)} 
                        placeholder="e.g. 102"
                      />
                    </div>
                  </div>
                </div>

                {/* Form Group 4: Lipid Profile */}
                <div className="form-card-section">
                  <div className="form-section-header">
                    <Sparkles size={15} color="#10b981" />
                    <span>Lipid Profile</span>
                  </div>
                  <div className="form-inputs-grid">
                    <div className="vital-field-box">
                      <label>Total Cholesterol (mg/dL)</label>
                      <input 
                        type="number" 
                        value={currentPatient.total_chol ?? ''} 
                        onChange={e => handleVitalChange('total_chol', e.target.value)} 
                        placeholder="e.g. 195"
                      />
                    </div>
                    <div className="vital-field-box">
                      <label>HDL Cholesterol (mg/dL)</label>
                      <input 
                        type="number" 
                        value={currentPatient.hdl ?? ''} 
                        onChange={e => handleVitalChange('hdl', e.target.value)} 
                        placeholder="e.g. 38"
                      />
                    </div>
                  </div>
                </div>

              </div>

              {/* Form Action Footer */}
              <div className="form-footer-action-bar">
                <button
                  className="btn-predict-now"
                  onClick={() => runDirectPrediction(currentPatient, task)}
                  disabled={predictionLoading}
                  aria-busy={predictionLoading}
                >
                  {predictionLoading ? (
                    <><Activity size={16} className="prediction-button-spinner" /> Calculating risk & XAI factors…</>
                  ) : (
                    <><Sparkles size={16} /> Predict Risk & Update XAI Factors</>
                  )}
                </button>
                <button 
                  className="btn-send-to-chat"
                  onClick={() => {
                    setInputMode('chat');
                    handleSendMessage(`Evaluate patient vitals: Age ${currentPatient.age}, HbA1c ${currentPatient.hba1c}%, BP ${currentPatient.sbp}/${currentPatient.dbp}, BMI ${currentPatient.bmi}, Insulin: ${currentPatient.insulin_use ? 'Yes' : 'No'}.`);
                  }}
                >
                  <Bot size={15} /> Explain in Chat Assistant
                </button>
              </div>
            </div>
          ) : (
            <div className="chat-view-container">
              <div 
                className="chat-history"
                ref={chatHistoryRef}
                onScroll={handleChatScroll}
              >
                {messages.map((m, idx) => (
                  <div key={idx} className={`chat-bubble-row ${m.role}`}>
                    <div className={`avatar-badge ${m.role}`}>
                      {m.role === 'assistant' ? <img src="/logo.png" alt="DKD-XAI" className="chat-avatar-logo" /> : <User size={18} />}
                    </div>
                    <div className={`chat-bubble ${m.role}`}>
                      {renderMessageContent(m.content)}
                      {m.isStreaming && <span className="typing-cursor">▊</span>}
                    </div>
                  </div>
                ))}

                {loading && (
                  <div className="chat-bubble-row assistant">
                    <div className="avatar-badge assistant">
                      <img src="/logo.png" alt="DKD-XAI" className="chat-avatar-logo" />
                    </div>
                    <div className="chat-bubble assistant">
                      <div className="typing-dots">
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                        <span className="typing-dot"></span>
                      </div>
                    </div>
                  </div>
                )}
              </div>

              {/* Chat Input Bar */}
              <div className="chat-input-area">
                <textarea
                  className="chat-textarea"
                  placeholder="Type patient details naturally (e.g., '62yo male, HbA1c 8.5%, BP 142/88, BMI 31, insulin 10 years...')"
                  value={inputMessage}
                  onChange={e => setInputMessage(e.target.value)}
                  onKeyDown={e => {
                    if (e.key === 'Enter' && !e.shiftKey) {
                      e.preventDefault();
                      handleSendMessage();
                    }
                  }}
                  rows={1}
                />
                <button 
                  className="send-btn" 
                  onClick={() => handleSendMessage()} 
                  disabled={loading || !inputMessage.trim()}
                >
                  <Send size={18} />
                </button>
              </div>
            </div>
          )}
        </main>

        {/* Right Column: Clean Risk Summary & SHAP Factors */}
        <aside className="panel-card right-summary-panel">
          <div className="panel-header">
            <div className="panel-title">
              <Sparkles size={16} color={userRole === 'doctor' ? '#0F766E' : '#2563EB'} />
              {userRole === 'doctor' ? 'Risk & Key Factors' : 'Your Kidney Health Summary'}
            </div>
            <span className="badge-model">
              {predictionLoading ? 'Calculating…' : userRole === 'doctor' ? (prediction ? prediction.model_name : 'XGBoost') : 'Personal Guide'}
            </span>
          </div>

          {predictionLoading && (
            <div className="prediction-loading-message" role="status">
              <Activity size={15} className="prediction-button-spinner" /> Calculating risk and key factors…
            </div>
          )}
          {predictionError && (
            <div className="prediction-error-message" role="alert">
              <AlertTriangle size={15} /> <span>{predictionError}</span>
            </div>
          )}

          <div className="right-panel-content">
            {/* Clean Risk Meter Card */}
            <div className="risk-meter-card">
              <div className="risk-top-row">
                <div>
                  <div className={`risk-number ${predictionLoading ? 'calculating' : ''}`} style={{ color: isHighRisk ? '#DC2626' : (userRole === 'doctor' ? '#0F766E' : '#2563EB') }}>
                    {prediction ? `${riskVal}%` : predictionLoading ? '…' : '—'}
                  </div>
                  <div className="risk-caption">
                    {predictionLoading ? 'Calculating risk' : predictionError ? 'Prediction unavailable' : userRole === 'doctor' ? 'Predicted DKD Risk' : 'Estimated Kidney Risk'}
                  </div>
                </div>

                <div className={`risk-status-pill ${predictionLoading || predictionError || !prediction ? 'pending' : isHighRisk ? 'high' : 'low'}`}>
                  {predictionLoading ? <Activity size={13} className="prediction-button-spinner" /> : predictionError ? <AlertTriangle size={13} /> : prediction ? (isHighRisk ? <AlertTriangle size={13} /> : <CheckCircle2 size={13} />) : <Activity size={13} />}
                  {predictionLoading ? 'CALCULATING' : predictionError ? 'UNAVAILABLE' : prediction ? (isHighRisk ? (userRole === 'doctor' ? 'HIGH RISK' : 'ELEVATED RISK') : 'LOW RISK') : 'AWAITING VITALS'}
                </div>
              </div>

              {/* Linear Progress Meter */}
              <div className="progress-meter-track">
                <div 
                  className="progress-meter-fill"
                  style={{ 
                    width: `${prediction ? Math.min(riskVal, 100) : 0}%`,
                    background: isHighRisk 
                      ? 'linear-gradient(90deg, #F59E0B, #DC2626)' 
                      : (userRole === 'doctor'
                          ? 'linear-gradient(90deg, #0F766E, #2563EB)'
                          : 'linear-gradient(90deg, #10B981, #2563EB)')
                  }}
                />
                {/* Threshold Marker */}
                <div 
                  className="threshold-marker" 
                  style={{ left: `${thresholdVal}%` }}
                  title={`Clinical Decision Threshold: ${thresholdVal}%`}
                />
              </div>

              <div className="meter-legend">
                <span>0% Safe</span>
                <span className="legend-threshold">Threshold: {thresholdVal}%</span>
                <span>100% Critical</span>
              </div>
            </div>

            {/* Active Patient Vitals Chips */}
            <div className="patient-chips-wrap">
              <div className="section-label">Current Active Patient Vitals</div>
              <div className="vitals-chips">
                <span className="vital-chip">Age: <strong>{currentPatient.age != null ? currentPatient.age : '--'}</strong></span>
                <span className="vital-chip">Sex: <strong>{currentPatient.female != null ? (currentPatient.female === 1 ? 'Female' : 'Male') : '--'}</strong></span>
                <span className="vital-chip">HbA1c: <strong>{currentPatient.hba1c != null ? `${currentPatient.hba1c}%` : '--'}</strong></span>
                <span className="vital-chip">BP: <strong>{currentPatient.sbp != null && currentPatient.dbp != null ? `${currentPatient.sbp}/${currentPatient.dbp}` : currentPatient.sbp != null ? `${currentPatient.sbp} mmHg` : '--'}</strong></span>
                <span className="vital-chip">BMI: <strong>{currentPatient.bmi != null ? currentPatient.bmi : '--'}</strong></span>
                <span className="vital-chip">Insulin: <strong>{currentPatient.insulin_use != null ? (currentPatient.insulin_use === 1 ? 'Yes' : 'No') : '--'}</strong></span>
                <span className="vital-chip">Duration: <strong>{currentPatient.dm_duration_years != null ? `${currentPatient.dm_duration_years}y` : '--'}</strong></span>
              </div>
            </div>

            {/* Top SHAP Drivers */}
            <div className="shap-factors-section">
              <div className="section-label">
                {userRole === 'doctor' ? 'Top Contributing Factors (SHAP)' : 'Key Health Factors & What They Mean'}
              </div>
              <div className="shap-card-list">
                {prediction && prediction.top_contributors ? (
                  prediction.top_contributors.map((c, i) => (
                    <div key={i} className={`shap-item-clean ${c.shap_value > 0 ? 'risk-up' : 'risk-down'}`}>
                      <div className="shap-item-header">
                        <span className="shap-feature-name">
                          {userRole === 'patient' && c.feature === 'hba1c' ? 'Blood Sugar (HbA1c)' :
                           userRole === 'patient' && c.feature === 'sbp' ? 'Blood Pressure (Systolic)' :
                           userRole === 'patient' && c.feature === 'dm_duration_years' ? 'Years with Diabetes' :
                           userRole === 'patient' && c.feature === 'insulin_use' ? 'Insulin Medication' :
                           userRole === 'patient' && c.feature === 'bmi' ? 'Body Weight / BMI' :
                           c.label} {c.value !== null && <span className="shap-feature-val">({c.value})</span>}
                        </span>
                        <span className={`shap-tag ${c.shap_value > 0 ? 'up' : 'down'}`}>
                          {userRole === 'patient'
                            ? (c.shap_value > 0 ? 'Adds Risk' : 'Protective')
                            : (c.shap_value > 0 ? `+${(c.shap_value * 100).toFixed(1)}%` : `${(c.shap_value * 100).toFixed(1)}%`)}
                        </span>
                      </div>
                      
                      {/* Visual magnitude indicator bar */}
                      <div className="shap-magnitude-track">
                        <div 
                          className={`shap-magnitude-fill ${c.shap_value > 0 ? 'up' : 'down'}`}
                          style={{ width: `${Math.min(Math.abs(c.shap_value * 100) * 1.5, 100)}%` }}
                        />
                      </div>

                      <div className="shap-evidence-quote">
                        {c.evidence_fact}
                      </div>

                      {c.evidence_url && (
                        <a 
                          href={c.evidence_url} 
                          target="_blank" 
                          rel="noreferrer" 
                          className="shap-citation-link"
                        >
                          Source: {c.evidence_source} <ExternalLink size={10} />
                        </a>
                      )}
                    </div>
                  ))
                ) : (
                  <div className="empty-notice">
                    {userRole === 'doctor'
                      ? 'Essential clinical markers (Age, HbA1c, Blood Pressure) are required to calculate risk and SHAP attribution. Enter vitals or select a benchmark case above.'
                      : 'Essential health numbers (Age, Blood Sugar HbA1c, Blood Pressure) are required to evaluate your kidney health. Enter your numbers or pick an example patient above.'}
                  </div>
                )}
              </div>
            </div>

            {/* Faithfulness Badge */}
            <div className="faithfulness-badge-clean">
              <ShieldCheck size={16} color="#10b981" />
              <span>
                {userRole === 'doctor'
                  ? 'Grounded in ADA 2026 & KDIGO Guidelines'
                  : 'Verified against Clinical Diabetes & Kidney Health Standards'}
              </span>
            </div>

          </div>
          {predictionAnimationKey > 0 && (
            <div
              key={predictionAnimationKey}
              className={`prediction-result-scan ${isHighRisk ? 'high' : 'low'}`}
              aria-hidden="true"
            />
          )}
        </aside>

      </div>
    </div>
  );
}
