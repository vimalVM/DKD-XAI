import React, { useState, useEffect, useRef } from 'react';
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
  ChevronDown,
  RotateCcw,
  SlidersHorizontal,
  X
} from 'lucide-react';

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

export default function App() {
  const [task, setTask] = useState('dkd_any');
  const [inputMode, setInputMode] = useState('form'); // 'form' | 'chat'
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: "Hello! I am your **DKD-XAI Clinical AI Assistant**.\n\nYou can describe a patient naturally in clinical terms, for example:\n> *\"71yo male with 16 years of diabetes, HbA1c 9.5%, BP 158/94, BMI 34, on insulin. What is his kidney risk?\"*\n\nOr pick any patient preset from the dropdown above to view an instant risk assessment.",
      prediction: null
    }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [showVitalsModal, setShowVitalsModal] = useState(false);
  const [currentPatient, setCurrentPatient] = useState({
    age: 68,
    female: 1,
    hba1c: 5.7,
    bmi: 38.9,
    waist_cm: 110,
    sbp: 134,
    dbp: 82,
    hdl: 29,
    total_chol: 193,
    dm_diagnosed: 1,
    insulin_use: 1,
    diabetes_pills: 0,
    dm_duration_years: 14
  });
  const [prediction, setPrediction] = useState(null);
  const [samples, setSamples] = useState(DEFAULT_SAMPLES);
  const [serverStatus, setServerStatus] = useState('connecting');
  const chatBottomRef = useRef(null);

  // Fetch samples & health
  useEffect(() => {
    fetch('/api/health')
      .then(res => res.json())
      .then(data => setServerStatus(data.status))
      .catch(() => setServerStatus('offline'));

    fetch('/api/sample-patients')
      .then(res => res.json())
      .then(data => setSamples(data))
      .catch(err => console.error("Could not load samples", err));
  }, []);

  // Run initial prediction
  useEffect(() => {
    runDirectPrediction(currentPatient, task);
  }, [task]);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const runDirectPrediction = (patientData, activeTask) => {
    fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ patient: patientData, task: activeTask })
    })
      .then(res => res.json())
      .then(data => setPrediction(data))
      .catch(err => console.error("Prediction failed", err));
  };

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

  const handleResetChat = () => {
    stopStreaming();
    setMessages([
      {
        role: 'assistant',
        content: "### Welcome to DKD-XAI Clinical AI Assistant\n\nDescribe a patient naturally in clinical terms, for example:\n> *\"71yo male with 16 years of diabetes, HbA1c 9.5%, BP 158/94, BMI 34, on insulin. What is his kidney risk?\"*\n\nOr pick any patient preset from the dropdown above to view an instant risk assessment.",
        prediction: null
      }
    ]);
  };

  const handleSendMessage = async (textToSend, patientOverride) => {
    const text = textToSend || inputMessage;
    if (!text.trim() || loading) return;

    stopStreaming();
    const patientToSend = patientOverride || currentPatient;
    const userMsg = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInputMessage('');
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: text,
          task: task,
          current_patient: patientToSend
        })
      });

      const data = await res.json();
      if (data.patient) {
        setCurrentPatient(prev => ({ ...prev, ...data.patient }));
      }
      if (data.prediction) {
        setPrediction(data.prediction);
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
    runDirectPrediction(found.data, task);
    handleSendMessage(`Analyze benchmark patient: ${found.name} (${found.description})`, found.data);
  };

  const handleVitalChange = (field, val) => {
    const updated = { ...currentPatient, [field]: val === '' ? null : Number(val) };
    setCurrentPatient(updated);
    runDirectPrediction(updated, task);
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

  return (
    <div className="app-container">
      {/* Top Navigation Bar */}
      <header className="app-header">
        <div className="brand-wrapper">
          <img src="/logo.png" alt="DKD-XAI Logo" className="brand-logo-img" />
          <div>
            <div className="brand-title">
              DKD-XAI <span className="brand-badge">Clinical Assistant</span>
            </div>
            <div className="brand-subtitle">
              Explainable AI for Early Diabetic Kidney Disease Risk
            </div>
          </div>
        </div>

        {/* Clean Controls in Header */}
        <div className="header-controls">
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
              Any Damage (XGBoost)
            </button>
            <button 
              className={`task-btn ${task === 'early_dkd' ? 'active' : ''}`}
              onClick={() => setTask('early_dkd')}
            >
              Early DKD vs None (MLP)
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
                <SlidersHorizontal size={14} /> Clinical Vitals Form
              </button>
              <button 
                className={`mode-toggle-btn ${inputMode === 'chat' ? 'active' : ''}`}
                onClick={() => setInputMode('chat')}
              >
                <Bot size={14} /> AI Chat Assistant
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
                >
                  <Sparkles size={16} /> Predict Risk & Update XAI Factors
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
              <div className="chat-history">
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
                <div ref={chatBottomRef} />
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
              <Sparkles size={16} color="#0F766E" /> Risk & Key Factors
            </div>
            <span className="badge-model">
              {prediction ? prediction.model_name : 'XGBoost'}
            </span>
          </div>

          <div className="right-panel-content">
            {/* Clean Risk Meter Card */}
            <div className="risk-meter-card">
              <div className="risk-top-row">
                <div>
                  <div className="risk-number" style={{ color: isHighRisk ? '#DC2626' : '#0F766E' }}>
                    {riskVal}%
                  </div>
                  <div className="risk-caption">Predicted DKD Risk</div>
                </div>

                <div className={`risk-status-pill ${isHighRisk ? 'high' : 'low'}`}>
                  {isHighRisk ? <AlertTriangle size={13} /> : <CheckCircle2 size={13} />}
                  {isHighRisk ? 'HIGH RISK' : 'LOW RISK'}
                </div>
              </div>

              {/* Linear Progress Meter */}
              <div className="progress-meter-track">
                <div 
                  className="progress-meter-fill"
                  style={{ 
                    width: `${Math.min(riskVal, 100)}%`,
                    background: isHighRisk 
                      ? 'linear-gradient(90deg, #F59E0B, #DC2626)' 
                      : 'linear-gradient(90deg, #0F766E, #2563EB)'
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
                <span className="vital-chip">Age: <strong>{currentPatient.age || '--'}</strong></span>
                <span className="vital-chip">Sex: <strong>{currentPatient.female === 1 ? 'Female' : 'Male'}</strong></span>
                <span className="vital-chip">HbA1c: <strong>{currentPatient.hba1c ? `${currentPatient.hba1c}%` : '--'}</strong></span>
                <span className="vital-chip">BP: <strong>{currentPatient.sbp && currentPatient.dbp ? `${currentPatient.sbp}/${currentPatient.dbp}` : '--'}</strong></span>
                <span className="vital-chip">BMI: <strong>{currentPatient.bmi || '--'}</strong></span>
                <span className="vital-chip">Insulin: <strong>{currentPatient.insulin_use ? 'Yes' : 'No'}</strong></span>
                <span className="vital-chip">Duration: <strong>{currentPatient.dm_duration_years ? `${currentPatient.dm_duration_years}y` : '--'}</strong></span>
              </div>
            </div>

            {/* Top SHAP Drivers */}
            <div className="shap-factors-section">
              <div className="section-label">Top Contributing Factors (SHAP)</div>
              <div className="shap-card-list">
                {prediction && prediction.top_contributors ? (
                  prediction.top_contributors.map((c, i) => (
                    <div key={i} className={`shap-item-clean ${c.shap_value > 0 ? 'risk-up' : 'risk-down'}`}>
                      <div className="shap-item-header">
                        <span className="shap-feature-name">
                          {c.label} {c.value !== null && <span className="shap-feature-val">({c.value})</span>}
                        </span>
                        <span className={`shap-tag ${c.shap_value > 0 ? 'up' : 'down'}`}>
                          {c.shap_value > 0 ? `+${(c.shap_value * 100).toFixed(1)}%` : `${(c.shap_value * 100).toFixed(1)}%`}
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
                    Chat with a patient profile to see live SHAP factors.
                  </div>
                )}
              </div>
            </div>

            {/* Faithfulness Badge */}
            <div className="faithfulness-badge-clean">
              <ShieldCheck size={16} color="#10b981" />
              <span>Grounded in ADA 2026 & KDIGO Guidelines</span>
            </div>

          </div>
        </aside>

      </div>
    </div>
  );
}
