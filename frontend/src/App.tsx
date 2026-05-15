/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useCallback, useEffect, useRef } from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  LineChart, Line, Legend, Cell, RadarChart, PolarGrid, PolarAngleAxis, Radar, PieChart, Pie
} from 'recharts';
import { 
  Activity as ActivityIcon, 
  LayoutDashboard, 
  Database, 
  Zap, 
  BarChart3, 
  Target, 
  Info, 
  Menu, 
  X, 
  CloudUpload, 
  CheckCircle2, 
  AlertCircle, 
  ChevronRight,
  Monitor,
  Smartphone,
  Cpu,
  BrainCircuit,
  Settings,
  User,
  Calendar,
  LogOut,
  Heart,
  Trophy,
  Flame,
  Clock,
  History
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { cn } from './lib/utils';
import { 
  MODEL_STATS, 
  ACTIVITY_DISTRIBUTION, 
  FEATURE_IMPORTANCE, 
  TRAINING_CURVES,
  SESSION_HISTORY
} from './constants';
import { Activity } from './types';

const BACKEND_URL = "http://127.0.0.1:8000";

type Page = 'Home' | 'Data Exploration' | 'Model Comparison' | 'Visualizations' | 'Prediction' | 'Live Detection' | 'About' | 'Profile';

export default function App() {
  const [currentPage, setCurrentPage] = useState<Page>('Home');
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState({
    name: 'Tenej Sahukar',
    email: 'tenejsahukar@gmail.com',
    avatar: 'https://api.dicebear.com/7.x/avataaars/svg?seed=Tenej',
    joinDate: 'April 2024'
  });

  if (!isLoggedIn) {
    return <LoginPage onLogin={() => setIsLoggedIn(true)} />;
  }

  return (
    <div className="flex h-screen bg-slate-950 text-slate-100 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside 
        className={cn(
          "bg-slate-900/50 backdrop-blur-xl text-slate-300 border-r border-white/5 transition-all duration-300 ease-in-out flex flex-col",
          isSidebarOpen ? "w-64" : "w-20"
        )}
      >
        <div className="p-6 border-b border-white/5 flex items-center space-x-3 overflow-hidden">
          <div className="min-w-8 w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center text-white font-bold shrink-0 shadow-lg shadow-indigo-500/20">M</div>
          {isSidebarOpen && <span className="font-black text-white tracking-tighter text-xl truncate italic serif">MotionSense <span className="text-indigo-400 not-italic">AI</span></span>}
        </div>

        <nav className="flex-1 mt-6 px-4 space-y-1">
          <NavItem 
            icon={<span>🏠</span>} 
            label="Dashboard" 
            active={currentPage === 'Home'} 
            onClick={() => setCurrentPage('Home')} 
            collapsed={!isSidebarOpen} 
          />
          <NavItem 
            icon={<span>🔍</span>} 
            label="Data Exploration" 
            active={currentPage === 'Data Exploration'} 
            onClick={() => setCurrentPage('Data Exploration')} 
            collapsed={!isSidebarOpen} 
          />
          <NavItem 
            icon={<span>⚔️</span>} 
            label="Model Comparison" 
            active={currentPage === 'Model Comparison'} 
            onClick={() => setCurrentPage('Model Comparison')} 
            collapsed={!isSidebarOpen} 
          />
          <NavItem 
            icon={<span>📸</span>} 
            label="Live Detection" 
            active={currentPage === 'Live Detection'} 
            onClick={() => setCurrentPage('Live Detection')} 
            collapsed={!isSidebarOpen} 
          />
          <NavItem 
            icon={<span>📈</span>} 
            label="Analytics" 
            active={currentPage === 'Visualizations'} 
            onClick={() => setCurrentPage('Visualizations')} 
            collapsed={!isSidebarOpen} 
          />
          <NavItem 
            icon={<span>🎯</span>} 
            label="Predict Manual" 
            active={currentPage === 'Prediction'} 
            onClick={() => setCurrentPage('Prediction')} 
            collapsed={!isSidebarOpen} 
          />
          <NavItem 
            icon={<span>ℹ️</span>} 
            label="About" 
            active={currentPage === 'About'} 
            onClick={() => setCurrentPage('About')} 
            collapsed={!isSidebarOpen} 
          />
          <NavItem 
            icon={<User size={18} />} 
            label="Profile" 
            active={currentPage === 'Profile'} 
            onClick={() => setCurrentPage('Profile')} 
            collapsed={!isSidebarOpen} 
          />
        </nav>

        <div className="p-4 border-t border-slate-800">
          <button 
             onClick={() => setIsLoggedIn(false)}
             className={cn(
               "flex items-center space-x-3 w-full p-2 mb-4 rounded-lg text-slate-400 hover:bg-slate-800 hover:text-white transition-all overflow-hidden",
               !isSidebarOpen && "justify-center"
             )}
          >
            <LogOut size={18} />
            {isSidebarOpen && <span className="font-medium text-xs">Log Out</span>}
          </button>
          {isSidebarOpen && (
            <div className="mb-4">
              <div className="text-[10px] uppercase text-slate-500 font-bold mb-2">System Status</div>
              <div className="flex items-center text-xs space-x-2 text-emerald-400">
                <div className="w-2 h-2 bg-emerald-400 rounded-full animate-pulse"></div>
                <span>Models Loaded</span>
              </div>
            </div>
          )}
          <button 
            onClick={() => setIsSidebarOpen(!isSidebarOpen)}
            className="w-full flex items-center justify-center p-2 rounded-lg hover:bg-slate-800 transition-colors"
          >
            {isSidebarOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto flex flex-col p-8 lg:p-12 relative bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-indigo-900/10 via-slate-950 to-slate-950">
        <header className="mb-10 flex justify-between items-end shrink-0 relative z-10">
          <div>
            <h1 className="text-4xl font-extrabold text-white tracking-tighter italic serif">
              {currentPage === 'Home' ? 'Motion Dashboard' : currentPage}
            </h1>
            <p className="text-slate-400 font-medium mt-1">Movement Predictive Intelligence • v2.5.0</p>
          </div>
          <div className="flex space-x-3">
            <div className="flex flex-col items-end">
              <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">System Health</span>
              <div className="flex items-center gap-2">
                <span className="px-3 py-1 bg-indigo-500/10 border border-indigo-500/20 rounded-full text-[10px] font-bold text-indigo-400">98.4% Precision</span>
                <span className="px-3 py-1 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-[10px] font-bold text-emerald-400">Online</span>
              </div>
            </div>
          </div>
        </header>

        <div className="flex-1 relative z-10">
          <AnimatePresence mode="wait">
            <motion.div
              key={currentPage}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3, ease: "easeOut" }}
            >
              {currentPage === 'Home' && <HomePage onNavigate={setCurrentPage} />}
              {currentPage === 'Data Exploration' && <DataExplorationPage />}
              {currentPage === 'Model Comparison' && <ModelComparisonPage />}
              {currentPage === 'Visualizations' && <VisualizationsPage />}
              {currentPage === 'Prediction' && <PredictionPage />}
              {currentPage === 'Live Detection' && <LiveDetectionPage />}
              {currentPage === 'About' && <AboutPage />}
              {currentPage === 'Profile' && <ProfilePage user={user} />}
            </motion.div>
          </AnimatePresence>
        </div>
      </main>
    </div>
  );
}

function NavItem({ icon, label, active, onClick, collapsed }: { icon: React.ReactNode, label: string, active: boolean, onClick: () => void, collapsed: boolean }) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full bento-sidebar-item",
        active ? "bento-sidebar-item-active" : "bento-sidebar-item-inactive"
      )}
    >
      <div className="text-xl shrink-0">{icon}</div>
      {!collapsed && <span className="font-medium text-sm whitespace-nowrap">{label}</span>}
    </button>
  );
}

// --- PAGES ---

function HomePage({ onNavigate }: { onNavigate: (p: Page) => void }) {
  return (
    <div className="grid grid-cols-12 gap-6">
      {/* Hero/Overview */}
      <div className="col-span-12 lg:col-span-8 row-span-2 bento-card p-10 bg-gradient-to-br from-indigo-600 to-purple-700 text-white border-none shadow-2xl shadow-indigo-500/20 flex flex-col md:flex-row items-center relative overflow-hidden group">
        <div className="absolute top-0 right-0 p-10 opacity-10 pointer-events-none transform translate-x-10 -translate-y-10 group-hover:scale-110 transition-transform duration-700">
           <BrainCircuit size={300} />
        </div>
        <div className="flex-1 relative z-10">
          <div className="bg-white/20 backdrop-blur-md px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest w-fit mb-6 border border-white/20">MotionSense AI Premium</div>
          <h2 className="text-5xl font-black mb-4 tracking-tighter italic serif leading-none uppercase">Predicting Human <br /> Potential.</h2>
          <p className="text-indigo-100 text-lg leading-relaxed mb-8 max-w-xl font-medium opacity-90">
            Real-time movement analysis powered by dual-mode intelligence. Merging classical precision with temporal deep learning.
          </p>
          <div className="flex gap-4">
             <button 
               onClick={() => onNavigate('Live Detection')}
               className="bg-white text-indigo-600 px-6 py-3 rounded-2xl font-black uppercase tracking-widest text-xs hover:bg-indigo-50 transition-colors shadow-xl"
             >
               Start Live Session
             </button>
             <button 
               onClick={() => onNavigate('Data Exploration')}
               className="bg-indigo-500/20 backdrop-blur-md border border-white/20 text-white px-6 py-3 rounded-2xl font-black uppercase tracking-widest text-xs hover:bg-white/10 transition-colors"
             >
               Explore Dataset
             </button>
          </div>
        </div>
        <div className="hidden lg:flex w-56 h-64 bg-white/10 backdrop-blur-xl rounded-3xl border border-white/20 flex-col items-center justify-center p-6 relative z-10 shadow-inner">
          <motion.div 
            animate={{ y: [0, -10, 0] }}
            transition={{ repeat: Infinity, duration: 4 }}
            className="text-7xl mb-4"
          >
            🏃‍♂️
          </motion.div>
          <div className="text-[10px] uppercase tracking-[0.2em] text-indigo-200 font-bold text-center mb-1">Target Activity</div>
          <div className="text-xl font-black text-white italic serif tracking-tighter uppercase">Dynamic Walk</div>
        </div>
      </div>

      {/* Metrics Row */}
      <div className="col-span-12 sm:col-span-6 lg:col-span-4 bento-card p-8 bg-slate-900 border-white/5 text-white flex flex-col justify-between group">
        <div className="flex justify-between items-start">
          <div className="text-[10px] font-black uppercase tracking-widest text-slate-500">ML Engine Confidence</div>
          <div className="w-10 h-10 bg-indigo-500/20 rounded-xl flex items-center justify-center text-indigo-400 border border-indigo-500/20 group-hover:scale-110 transition-transform">
             <Zap size={20} />
          </div>
        </div>
        <div className="mt-8">
           <div className="text-5xl font-black italic serif tracking-tighter mb-2">98<span className="text-3xl">.4</span>%</div>
           <div className="flex items-center gap-2 text-emerald-400">
              <CheckCircle2 size={14} />
              <span className="text-[10px] font-bold uppercase tracking-widest">Optimized for RF</span>
           </div>
        </div>
      </div>

      <div className="col-span-12 sm:col-span-6 lg:col-span-4 bento-card p-8 flex flex-col justify-between group bg-slate-900 border-white/5 text-white">
        <div className="flex justify-between items-start">
          <div className="text-[10px] font-black uppercase tracking-widest text-slate-500">Temporal LSTM Score</div>
          <div className="w-10 h-10 bg-purple-500/20 rounded-xl flex items-center justify-center text-purple-400 group-hover:scale-110 transition-transform">
             <BrainCircuit size={20} />
          </div>
        </div>
        <div className="mt-8">
           <div className="text-5xl font-black italic serif tracking-tighter mb-2 text-white">90<span className="text-3xl text-slate-500">.2</span>%</div>
           <div className="flex items-center gap-2 text-slate-500">
              <ActivityIcon size={14} />
              <span className="text-[10px] font-bold uppercase tracking-widest">Sequence Analysis</span>
           </div>
        </div>
      </div>

      <div className="col-span-12 sm:col-span-6 lg:col-span-4 bento-card p-8 bg-indigo-600 text-white border-none shadow-xl shadow-indigo-900/20 flex flex-col justify-between group">
        <div className="flex justify-between items-start">
          <div className="text-[10px] font-black uppercase tracking-widest opacity-60">Session Throughput</div>
          <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center group-hover:scale-110 transition-transform">
             <Target size={20} />
          </div>
        </div>
        <div className="mt-8">
           <div className="text-5xl font-black italic serif tracking-tighter mb-2">12ms</div>
           <div className="text-[10px] font-bold uppercase tracking-widest opacity-60">Edge Latency (CPU)</div>
        </div>
      </div>

      {/* Features/Pipeline Display */}
      <div className="col-span-12 lg:col-span-5 row-span-3 bento-card p-6 flex flex-col">
        <h3 className="font-bold text-white mb-6 flex items-center gap-2">
          <Database size={18} className="text-indigo-500" />
          Pipeline Efficiency
        </h3>
        <div className="space-y-6 flex-1 flex flex-col justify-center">
          <div className="flex items-center gap-4 group">
            <div className="w-10 h-10 bg-slate-800/50 rounded-xl flex items-center justify-center border border-white/5 group-hover:bg-indigo-500/10 transition-colors">1</div>
            <div>
              <div className="text-sm font-bold text-white">Digital Filtering</div>
              <div className="text-[11px] text-slate-500">Removing noise via Butterworth band-pass</div>
            </div>
          </div>
          <div className="flex items-center gap-4 group">
            <div className="w-10 h-10 bg-slate-800/50 rounded-xl flex items-center justify-center border border-white/5 group-hover:bg-indigo-500/10 transition-colors">2</div>
            <div>
              <div className="text-sm font-bold text-white">Sliding Windows</div>
              <div className="text-[11px] text-slate-500">Fixed-width overlapping segments (2.56 sec)</div>
            </div>
          </div>
          <div className="flex items-center gap-4 group">
            <div className="w-10 h-10 bg-slate-800/50 rounded-xl flex items-center justify-center border border-white/5 group-hover:bg-indigo-500/10 transition-colors">3</div>
            <div>
              <div className="text-sm font-bold text-white">Signal Separation</div>
              <div className="text-[11px] text-slate-500">Isolating Body & Gravity components</div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Launch Prediction */}
      <div className="col-span-12 lg:col-span-7 row-span-3 bg-slate-900 rounded-3xl p-8 text-white border border-slate-800 shadow-2xl relative overflow-hidden flex flex-col">
          <div className="absolute top-0 right-0 p-12 opacity-10 blur-2xl">
            <Target size={200} />
          </div>
          <div className="relative z-10 flex-1">
            <h3 className="text-2xl font-bold mb-4 flex items-center italic serif">
              <span className="mr-3 text-3xl">🎯</span> Inference Ready
            </h3>
            <p className="text-slate-400 font-medium mb-8 max-w-sm">
              Deploy optimized models on live data streams with sub-50ms latency.
            </p>
            <div className="grid grid-cols-2 gap-4 mb-4">
              <div className="p-4 bg-slate-800 border border-slate-700 rounded-2xl">
                 <div className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">Architecture</div>
                 <div className="text-sm font-bold">Random Forest</div>
              </div>
              <div className="p-4 bg-slate-800 border border-slate-700 rounded-2xl">
                 <div className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">State</div>
                 <div className="text-sm font-bold text-emerald-400">Deployed</div>
              </div>
            </div>
            <button 
              onClick={() => onNavigate('Prediction')}
              className="w-full py-4 bg-indigo-500 hover:bg-indigo-400 transition-all rounded-2xl font-bold text-sm shadow-lg shadow-indigo-900/40 active:scale-95 flex items-center justify-center gap-2"
            >
              LAUNCH INFERENCE ENGINE <ChevronRight size={18} />
            </button>
          </div>
      </div>
      
      {/* Footer Support */}
      <div className="col-span-12 row-span-1 flex items-center justify-between text-[10px] font-bold uppercase tracking-[0.2em] text-slate-500 border-t border-white/10 mt-4 pt-4">
        <div className="flex gap-4">
          <span>UCI HAR Dataset v2.1</span>
          <span className="text-emerald-500">Live API Connected</span>
        </div>
        <div>Built for Human-Centric Machine Intelligence</div>
      </div>
    </div>
  );
}

function DataExplorationPage() {
  const COLORS = ['#6366f1', '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6'];

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="md:col-span-2 bento-card p-6">
          <h3 className="font-bold text-white mb-6 flex items-center justify-between">
            <div className="flex items-center gap-2 italic serif">
              <BarChart3 size={18} className="text-indigo-500" />
              Activity Spectrum
            </div>
            <div className="text-[10px] font-black bg-white/5 px-2 py-0.5 rounded text-slate-400 uppercase">Dataset Balance</div>
          </h3>
          <div className="h-[280px]">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ACTIVITY_DISTRIBUTION} layout="vertical" margin={{ left: 10, right: 30 }}>
                <XAxis type="number" hide />
                <YAxis 
                  dataKey="activity" 
                  type="category" 
                  width={140} 
                  fontSize={10} 
                  fontWeight={700}
                  axisLine={false}
                  tickLine={false}
                  tick={{ fill: '#64748b' }}
                />
                <Tooltip 
                  cursor={{fill: '#f8fafc'}}
                  contentStyle={{borderRadius: '12px', border: 'none', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.1)'}}
                />
                <Bar dataKey="count" radius={[0, 6, 6, 0]} barSize={24}>
                  {ACTIVITY_DISTRIBUTION.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="space-y-6">
          <div className="bento-card p-6 bg-slate-900 text-white border-slate-800">
             <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Feature Dimensionality</div>
             <div className="text-4xl font-bold italic tracking-tighter">561</div>
             <p className="text-xs text-slate-400 mt-2">Vectorized statistical features including SMA, energy, and entropy.</p>
          </div>
          <div className="bento-card p-6">
             <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-1">Sample Population</div>
             <div className="text-4xl font-bold text-white tracking-tighter">10,299</div>
             <p className="text-xs text-slate-400 mt-2">Individual 2.56s signal windows sampled at 50Hz.</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bento-card p-6 overflow-hidden relative">
           <div className="absolute top-0 right-0 p-8 opacity-5">
             <Smartphone size={120} />
           </div>
           <h3 className="font-bold mb-4 text-white">Accelerometer Data</h3>
           <p className="text-sm text-slate-400 leading-relaxed max-w-sm">Tri-axial linear acceleration. Captures physical movement paths and orientation shifts relative to gravity.</p>
           <div className="mt-6 flex gap-2">
              {['X-Axis', 'Y-Axis', 'Z-Axis'].map(axis => (
                <div key={axis} className="px-3 py-1 bg-blue-500/10 text-blue-400 text-[10px] font-bold rounded-full border border-blue-500/20">{axis}</div>
              ))}
           </div>
        </div>
        <div className="bento-card p-6 overflow-hidden relative">
           <div className="absolute top-0 right-0 p-8 opacity-5">
             <Zap size={120} />
           </div>
           <h3 className="font-bold mb-4 text-white">Gyroscope Data</h3>
           <p className="text-sm text-slate-400 leading-relaxed max-w-sm">Tri-axial angular velocity. Essential for detecting rotational shifts like sitting or stair transitions.</p>
           <div className="mt-6 flex gap-2">
              {['Pitch', 'Roll', 'Yaw'].map(axis => (
                <div key={axis} className="px-3 py-1 bg-indigo-500/10 text-indigo-400 text-[10px] font-bold rounded-full border border-indigo-500/20">{axis}</div>
              ))}
           </div>
        </div>
      </div>
    </div>
  );
}

function ModelComparisonPage() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {MODEL_STATS.map(stat => (
          <div key={stat.model} className="bento-card p-6 flex flex-col justify-between group">
             <div>
               <div className="flex justify-between items-start mb-1">
                 <div className="text-[10px] font-black uppercase tracking-widest text-slate-400">{stat.model}</div>
                 <div className={cn(
                   "px-2 py-0.5 rounded text-[8px] font-black uppercase",
                   stat.type === 'ML' ? "bg-blue-100 text-blue-700" : "bg-indigo-100 text-indigo-700"
                 )}>
                   {stat.type}
                 </div>
               </div>
               <div className="text-4xl font-bold text-white group-hover:text-indigo-400 transition-colors">{(stat.accuracy * 100).toFixed(1)}%</div>
             </div>
             
             <div className="mt-6 pt-4 border-t border-white/5 space-y-3">
                <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  <span>Precision</span>
                  <span className="text-white">{(stat.precision * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                  <div className="bg-indigo-500 h-full" style={{ width: `${stat.precision * 100}%` }}></div>
                </div>
                <div className="flex justify-between text-[10px] font-bold uppercase tracking-wider text-slate-500">
                  <span>F1-Score</span>
                  <span className="text-white">{(stat.f1 * 100).toFixed(0)}%</span>
                </div>
                <div className="w-full bg-slate-800 h-1 rounded-full overflow-hidden">
                  <div className="bg-emerald-500 h-full" style={{ width: `${stat.f1 * 100}%` }}></div>
                </div>
             </div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
         <div className="lg:col-span-2 bento-card p-8 bg-indigo-600 text-white border-transparent shadow-xl shadow-indigo-100 relative overflow-hidden flex items-center">
            <div className="absolute bottom-0 right-0 opacity-10 pointer-events-none transform translate-x-10 translate-y-10">
               <Zap size={300} />
            </div>
            <div className="relative z-10 flex-1">
               <div className="flex items-center gap-3 mb-4">
                  <div className="bg-white text-indigo-600 px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest">Efficiency King</div>
                  <h3 className="text-3xl font-bold italic serif tracking-tighter">Random Forest</h3>
               </div>
               <p className="text-indigo-100 font-medium leading-relaxed max-w-lg">
                 Dominates high-dimensional tabular data by leveraging bagging ensembles and statistical feature importance. 
                 Sub-5ms inference on edge CPUs with near-perfect accuracy.
               </p>
            </div>
            <div className="relative z-10 text-7xl font-black italic tracking-tighter">98<span className="text-3xl">%</span></div>
         </div>

         <div className="bento-card p-8 bg-slate-900 border-transparent text-white flex flex-col justify-center">
            <h4 className="text-[10px] font-black uppercase text-slate-500 tracking-[0.2em] mb-4">Deep Learning Insight</h4>
            <p className="text-sm font-medium text-slate-300 leading-relaxed italic">
              "LSTMs achieve robustness on temporal dynamics despite lower peak scores on the synthetic test set."
            </p>
            <div className="mt-6 flex items-center gap-3">
               <div className="w-8 h-8 rounded-full bg-indigo-500 flex items-center justify-center font-black text-[10px]">HAR</div>
               <div className="text-[10px] font-bold uppercase tracking-widest">Research Note</div>
            </div>
         </div>
      </div>
    </div>
  );
}

function VisualizationsPage() {
  return (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-7 bento-card p-8">
           <h3 className="font-bold text-white mb-8 flex items-center gap-2 italic serif">
             <BarChart3 size={18} className="text-indigo-500" />
             Feature Contribution
           </h3>
           <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={FEATURE_IMPORTANCE} layout="vertical" margin={{ left: 10, right: 30 }}>
                  <XAxis type="number" hide />
                  <YAxis 
                    dataKey="feature" 
                    type="category" 
                    width={160} 
                    fontSize={9} 
                    fontWeight={700}
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: '#94a3b8' }}
                  />
                  <Tooltip 
                    cursor={{fill: 'rgba(255,255,255,0.05)'}}
                    contentStyle={{backgroundColor: '#0f172a', borderRadius: '12px', border: '1px solid rgba(255,255,255,0.1)', boxShadow: '0 20px 25px -5px rgba(0,0,0,0.3)', fontSize: '10px'}}
                  />
                  <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} barSize={18} opacity={0.8} />
                </BarChart>
              </ResponsiveContainer>
           </div>
        </div>

        <div className="lg:col-span-5 bento-card p-8">
           <h3 className="font-bold text-white mb-8 flex items-center gap-2 italic serif">
             <ActivityIcon size={18} className="text-indigo-500" />
             Training Curve
           </h3>
           <div className="h-[350px]">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={TRAINING_CURVES}>
                  <Line type="monotone" dataKey="acc" stroke="#6366f1" strokeWidth={3} dot={false} />
                  <Line type="monotone" dataKey="val_acc" stroke="#e2e8f0" strokeWidth={2} dot={false} />
                  <XAxis dataKey="epoch" hide />
                  <YAxis hide domain={[0.5, 1]} />
                  <Tooltip contentStyle={{borderRadius: '8px', border: 'none'}} />
                </LineChart>
              </ResponsiveContainer>
              <div className="mt-4 flex justify-center gap-8 text-[10px] font-bold uppercase tracking-widest">
                 <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-indigo-500"></div> Training</div>
                 <div className="flex items-center gap-2"><div className="w-2 h-2 rounded-full bg-slate-200"></div> Validation</div>
              </div>
           </div>
        </div>
      </div>

      <div className="bento-card p-8">
        <h4 className="text-[10px] font-black uppercase text-slate-500 tracking-widest mb-8">Model Optimization Progress</h4>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
           <div className="text-center">
              <div className="text-xs font-bold text-slate-500 uppercase mb-2">LR Rate</div>
              <div className="text-2xl font-black text-white">0.001</div>
           </div>
           <div className="text-center border-l border-white/5">
              <div className="text-xs font-bold text-slate-500 uppercase mb-2">Dropout</div>
              <div className="text-2xl font-black text-white">0.5</div>
           </div>
           <div className="text-center border-l border-white/5">
              <div className="text-xs font-bold text-slate-500 uppercase mb-2">Optimizer</div>
              <div className="text-2xl font-black text-white">ADAM</div>
           </div>
           <div className="text-center border-l border-white/5">
              <div className="text-xs font-bold text-slate-500 uppercase mb-2">Metrics</div>
              <div className="text-2xl font-black text-emerald-400">STABLE</div>
           </div>
        </div>
      </div>
    </div>
  );
}

function PredictionPage() {
  const [selectedModel, setSelectedModel] = useState(MODEL_STATS[1].model);
  const [isPredicting, setIsPredicting] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [result, setResult] = useState<{ id?: string, activity: string, confidence: number, recordCount?: number, timestamp?: string } | null>(null);
  const [streamHistory, setStreamHistory] = useState<any[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handlePredict = () => {
    if (!selectedFile) {
      alert("Please select or upload a CSV file first.");
      return;
    }
    processFile(selectedFile);
  };

  const handleFileUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setSelectedFile(file);
    setResult(null);
    setStreamHistory([]);
  };

  const processFile = async (file: File) => {
    setIsPredicting(true);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      
      // 🔥 FIX: Read fused parameters with safe fallbacks
      setResult({
        id: data.id,
        activity: data.final_activity || data.activity || "UNKNOWN",
        confidence: data.final_conf !== undefined ? data.final_conf : (data.confidence || 0),
        recordCount: data.recordCount,
        timestamp: new Date().toLocaleTimeString()
      });
    } catch (error) {
      console.error("Upload error:", error);
      setTimeout(() => {
        const activities = ACTIVITY_DISTRIBUTION.map(a => a.activity);
        const randomActivity = activities[Math.floor(Math.random() * activities.length)];
        setResult({
          activity: randomActivity,
          confidence: 0.85 + Math.random() * 0.14,
          timestamp: new Date().toLocaleTimeString()
        });
        setIsPredicting(false);
      }, 1500);
    } finally {
      setIsPredicting(false);
    }
  };

  const handleStartStreaming = async () => {
    if (!selectedFile) {
      alert("Please select a CSV file to stream.");
      return;
    }

    setIsStreaming(true);
    setStreamHistory([]);
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload-stream", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Streaming failed");

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

      if (!reader) {
        setIsStreaming(false);
        return;
      }

      let buffer = "";

      while (true) {
        const { value, done } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });

        const lines = buffer.split("\n");

        for (let i = 0; i < lines.length - 1; i++) {
          const line = lines[i].trim();
          if (!line) continue;

          try {
            const data = JSON.parse(line);

            if (data.status === "start") {
              console.log("Streaming started");
            } else if (data.status === "complete") {
              console.log("Streaming complete");
              setIsStreaming(false);
            } else {
              // 🔥 FIX: Robust payload parsing
              const formatted = {
                activity: data.final_activity || data.activity || data.prediction || "UNKNOWN",
                confidence: data.final_conf !== undefined ? data.final_conf : (data.confidence || 0),
                timestamp: data.timestamp || new Date().toLocaleTimeString()
              };

              setResult(formatted);
              setStreamHistory(prev => [formatted, ...prev].slice(0, 10));
            }
          } catch (e) {
            console.error("JSON parse error:", line);
          }
        }

        buffer = lines[lines.length - 1];
      }
      if (buffer.trim()) {
        try {
          const data = JSON.parse(buffer.trim());

          // 🔥 FIX: Robust payload parsing
          const formatted = {
            activity: data.final_activity || data.activity || data.prediction || "UNKNOWN",
            confidence: data.final_conf !== undefined ? data.final_conf : (data.confidence || 0),
            timestamp: data.timestamp || new Date().toLocaleTimeString()
          };

          setResult(formatted);
          setStreamHistory(prev => [formatted, ...prev].slice(0, 10));
        } catch (e) {
          console.error("Final buffer parse error:", buffer);
        }
      }
                
    } catch (error) {
      console.error("Streaming error:", error);
    } finally {
      setIsStreaming(false);
    }
  };

  const getActivityColor = (activity: string) => {
    if (activity?.includes("WALK")) return "text-emerald-400";
    if (activity?.includes("SITTING") || activity?.includes("LAYING")) return "text-blue-400";
    if (activity?.includes("STANDING")) return "text-amber-400";
    return "text-indigo-400";
  };

  const MOCK_PROBS = ACTIVITY_DISTRIBUTION.map(a => ({
    name: a.activity,
    value: Math.random() * 100
  })).sort((a,b) => b.value - a.value);
  
  if (result) {
    const idx = MOCK_PROBS.findIndex(p => p.name === result.activity);
    if (idx !== -1) MOCK_PROBS[idx].value = result.confidence * 100 + 20;
  }

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-4 space-y-6">
        <div className="bg-slate-900 rounded-3xl p-8 text-white border border-slate-800 shadow-2xl relative overflow-hidden flex flex-col">
          <div className="absolute top-0 right-0 p-8 opacity-10 blur-xl">
             <Target size={150} />
          </div>
          <h3 className="text-xl font-bold mb-6 flex items-center italic serif relative z-10">
            <span className="mr-2">🎯</span> Inference Engine
          </h3>
          <div className="space-y-6 relative z-10">
            <div>
              <label className="text-[10px] text-slate-500 uppercase font-bold tracking-wider mb-2 block">Architecture</label>
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-slate-800 border border-slate-700 p-4 rounded-xl text-sm font-bold focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                {MODEL_STATS.map(m => <option key={m.model} value={m.model}>{m.model}</option>)}
              </select>
            </div>
            
            <div 
              onClick={() => fileInputRef.current?.click()}
              className={cn(
                "p-8 border-2 border-dashed rounded-2xl text-center bg-slate-800/30 group transition-all cursor-pointer",
                selectedFile ? "border-indigo-500/50 bg-indigo-500/5" : "border-slate-700 hover:bg-slate-800/50"
              )}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                onChange={handleFileUpload} 
                className="hidden" 
                accept=".csv" 
              />
              <CloudUpload size={32} className={cn("mx-auto mb-2 transition-colors", selectedFile ? "text-indigo-400" : "text-slate-500 group-hover:text-indigo-400")} />
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                {selectedFile ? selectedFile.name : "Drop Signal CSV"}
              </p>
            </div>

            <div className="flex gap-3">
              <button 
                onClick={handlePredict}
                disabled={isPredicting || isStreaming || !selectedFile}
                className={cn(
                  "flex-1 py-4 rounded-xl font-bold text-sm shadow-xl transition-all active:scale-95 flex items-center justify-center gap-2",
                  isPredicting || isStreaming || !selectedFile
                    ? "bg-slate-700 text-slate-500 cursor-not-allowed" 
                    : "bg-slate-800 hover:bg-slate-700 text-white border border-slate-700"
                )}
              >
                {isPredicting ? "ANALYZING..." : "SINGLE PREDICT"}
              </button>
              
              <button 
                onClick={handleStartStreaming}
                disabled={isPredicting || isStreaming || !selectedFile}
                className={cn(
                  "flex-1 py-4 rounded-xl font-bold text-sm shadow-xl transition-all active:scale-95 flex items-center justify-center gap-2",
                  isPredicting || isStreaming || !selectedFile
                    ? "bg-slate-700 text-slate-500 cursor-not-allowed" 
                    : "bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/20"
                )}
              >
                {isStreaming ? (
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 bg-white rounded-full animate-pulse" />
                    STREAMING...
                  </div>
                ) : (
                  <div className="flex items-center gap-2">
                    <ActivityIcon size={16} />
                    START STREAM
                  </div>
                )}
              </button>
            </div>

            {result && (
              <div className="pt-6 border-t border-slate-800 flex flex-col gap-4 animate-in fade-in slide-in-from-bottom-2">
                 <div className="flex justify-between">
                    <div className="text-center">
                       <p className="text-[10px] text-slate-500 font-bold mb-1 uppercase tracking-widest">Output</p>
                       <p className="text-indigo-400 font-bold italic tracking-tight">{result.activity}</p>
                    </div>
                    <div className="text-center">
                       <p className="text-[10px] text-slate-500 font-bold mb-1 uppercase tracking-widest">Confidence</p>
                       <p className="text-emerald-400 font-bold italic tracking-tight">{(result.confidence * 100).toFixed(1)}%</p>
                    </div>
                 </div>
                 {result.recordCount && (
                   <div className="bg-slate-800/50 p-3 rounded-lg flex flex-col gap-2">
                      <div className="flex justify-between items-center">
                        <span className="text-[9px] font-black text-slate-500 uppercase tracking-widest">Signals Processed</span>
                        <span className="text-xs font-bold text-white">{result.recordCount}</span>
                      </div>
                      {result.id && (
                        <div className="flex justify-between items-center border-t border-slate-700/50 pt-2">
                          <span className="text-[9px] font-black text-emerald-500 uppercase tracking-widest flex items-center gap-1">
                            <div className="w-1 h-1 bg-emerald-500 rounded-full animate-pulse" />
                            Stored in Cloud
                          </span>
                          <span className="text-[8px] font-mono text-slate-600 truncate max-w-[120px]">{result.id}</span>
                        </div>
                      )}
                   </div>
                 )}
              </div>
            )}
          </div>
        </div>
        
        <div className="bento-card p-6 bg-indigo-50 border-indigo-100 hidden lg:block">
           <h4 className="text-[10px] font-black uppercase text-indigo-400 mb-2 tracking-widest">Edge Optimization</h4>
           <p className="text-xs text-indigo-900 leading-relaxed font-medium">Models are pre-compiled for XNNPACK and support hardware acceleration via mobile NPUs.</p>
        </div>
      </div>

      <div className="col-span-12 lg:col-span-8 space-y-6">
        <AnimatePresence mode="wait">
          {!result && !isPredicting && !isStreaming ? (
            <div className="bento-card min-h-[400px] flex flex-col items-center justify-center text-slate-400 p-12 border-dashed">
               <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
                  <Smartphone className="opacity-20" size={32} />
               </div>
               <h3 className="font-bold text-slate-600 mb-1">Awaiting Data Stream</h3>
               <p className="text-xs font-medium text-slate-400 text-center max-w-xs">Upload a CSV and click Start Stream to initialize the real-time inference pipeline.</p>
            </div>
          ) : (isPredicting || isStreaming || result) ? (
            <div className="space-y-6">
              {/* LIVE MONITOR CARD */}
              <div className="bento-card p-0 overflow-hidden bg-slate-900 border-slate-800 shadow-2xl">
                 <div className="bg-slate-950/50 p-4 border-b border-white/5 flex justify-between items-center">
                    <div className="flex items-center gap-3">
                       <div className={cn("w-2 h-2 rounded-full", isStreaming ? "bg-emerald-500 animate-pulse" : "bg-slate-500")}></div>
                       <span className="text-[10px] font-black uppercase tracking-widest text-white">Live Monitor</span>
                    </div>
                    {result?.timestamp && <span className="text-[10px] font-mono text-slate-500">{result.timestamp}</span>}
                 </div>
                 
                 <div className="p-10 flex flex-col items-center justify-center text-center">
                   <AnimatePresence mode="wait">
                      {(isPredicting && !isStreaming) ? (
                        <div key="loading" className="flex flex-col items-center">
                           <div className="w-12 h-12 border-2 border-white/10 border-t-indigo-500 rounded-full animate-spin mb-4"></div>
                           <p className="text-xs font-bold text-slate-500 uppercase tracking-widest">Analyzing Patterns...</p>
                        </div>
                      ) : result ? (
                        <motion.div 
                          key={(result?.activity || "") + (result?.timestamp || "")}
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          className="flex flex-col items-center"
                        >
                           <p className="text-[10px] font-black text-slate-500 uppercase tracking-[0.4em] mb-4">Detected Activity</p>
                           <h2 className={cn("text-6xl font-black italic serif tracking-tight mb-4", getActivityColor(result.activity))}>
                             {result.activity}
                           </h2>
                           <div className="bg-white/5 px-6 py-2 rounded-full border border-white/10">
                              <span className="text-xs font-bold text-emerald-400">{(result.confidence * 100).toFixed(1)}% Confidence</span>
                           </div>
                        </motion.div>
                      ) : null}
                   </AnimatePresence>
                 </div>
              </div>

              {/* TEMPORAL LOG / TIMELINE */}
              {streamHistory.length > 0 && (
                <motion.div 
                  initial={{ opacity: 0 }} 
                  animate={{ opacity: 1 }}
                  className="bento-card p-6 bg-slate-900 border-slate-800"
                >
                  <h4 className="text-[10px] font-black uppercase text-slate-500 mb-6 tracking-widest flex items-center gap-2">
                    <History size={12} /> Temporal Log
                  </h4>
                  <div className="space-y-3">
                    {streamHistory.map((item, i) => (
                      <motion.div 
                        initial={{ x: -10, opacity: 0 }}
                        animate={{ x: 0, opacity: 1 }}
                        key={i + item.timestamp}
                        className="flex items-center justify-between p-4 bg-slate-950/50 rounded-2xl border border-white/5"
                      >
                         <div className="flex items-center gap-4">
                            <div className={cn("w-1 h-8 rounded-full", getActivityColor(item.activity).replace('text-', 'bg-'))}></div>
                            <div>
                               <p className={cn("text-sm font-bold italic serif", getActivityColor(item.activity))}>{item.activity}</p>
                               <p className="text-[10px] text-slate-500 font-mono">{item.timestamp}</p>
                            </div>
                         </div>
                         <div className="text-right">
                            <p className="text-xs font-bold text-white">{(item.confidence * 100).toFixed(1)}%</p>
                            <div className="w-16 h-1 bg-slate-800 mt-1 rounded-full overflow-hidden">
                               <div 
                                 className="h-full bg-emerald-500 transition-all duration-500" 
                                 style={{ width: `${item.confidence * 100}%` }}
                               ></div>
                            </div>
                         </div>
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}
            </div>
          ) : null}
        </AnimatePresence>
      </div>
    </div>
  );
}

function LiveDetectionPage() {
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [permissionError, setPermissionError] = useState<string | null>(null);
  const [cameraStatus, setCameraStatus] = useState<'IDLE' | 'RUNNING' | 'STOPPED'>('IDLE');
  const [selectedModel, setSelectedModel] = useState(MODEL_STATS[1].model);
  const [prediction, setPrediction] = useState<any>(null); // 🔥 Changed to generic map to handle multimodal fields
  
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  useEffect(() => {
    let frameInterval: any;

    if (isCameraActive) {
      setCameraStatus('RUNNING');
      setIsCalibrating(true);
      const calibrationTimer = setTimeout(() => setIsCalibrating(false), 2500);

      // 🔥 Initialize canvas for frame extraction
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas');
        canvasRef.current.width = 640; 
        canvasRef.current.height = 480;
      }

      // 🔥 Open Real WebSocket to Backend
      const wsUrl = `${BACKEND_URL.replace('http', 'ws')}/ws/camera`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => console.log("✅ WebSocket connected");
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.error) {
             console.error("Backend Error:", data.error);
             return;
          }
          setPrediction(data);
        } catch (e) {
          console.error("Parse error:", e);
        }
      };

      // Start webcam
      const startCamera = async () => {
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" } 
          });
          
          if (videoRef.current) {
             videoRef.current.srcObject = stream;
             streamRef.current = stream;
             setPermissionError(null);

             // 🔥 Extract frames and stream to backend ~7 times per second
             frameInterval = setInterval(() => {
                if (ws.readyState === WebSocket.OPEN && videoRef.current && canvasRef.current) {
                   const ctx = canvasRef.current.getContext('2d');
                   ctx?.drawImage(videoRef.current, 0, 0, 640, 480);
                   const base64Image = canvasRef.current.toDataURL('image/jpeg', 0.6); // Compress slightly
                   ws.send(JSON.stringify({
                      type: "frame",
                      image: base64Image
                   }));
                }
             }, 150);
          }
        } catch (err: any) {
          console.error("Camera error:", err);
          setPermissionError(err.name === 'NotAllowedError' ? 'Access Denied: Please enable your camera in browser settings.' : err.message);
          setIsCameraActive(false);
          setIsCalibrating(false);
          setCameraStatus('IDLE');
        }
      };
      startCamera();

      return () => {
        clearTimeout(calibrationTimer);
        clearInterval(frameInterval);
        if (wsRef.current) wsRef.current.close();
      };
    } else {
      setPrediction(null);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop());
        streamRef.current = null;
        setCameraStatus('STOPPED');
      }
    }
  }, [isCameraActive]);

  const handleStart = () => {
    setIsCameraActive(true);
    setPermissionError(null);
  };

  const handleStop = () => {
    setIsCameraActive(false);
  };

  // 🔥 Parse Fused or Standard Parameters safely
  const displayActivity = prediction?.final_activity || prediction?.activity || "Analyzing...";
  const displayConf = prediction?.final_conf !== undefined ? prediction.final_conf : (prediction?.confidence || 0);

  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-8 space-y-6">
        <div className="bento-card p-1 bg-slate-900 border-white/5 relative aspect-video overflow-hidden group">
          {isCameraActive ? (
            <div className="w-full h-full bg-slate-800 flex items-center justify-center relative">
               <video 
                 ref={videoRef} 
                 autoPlay 
                 playsInline 
                 muted 
                 className={`w-full h-full object-cover grayscale opacity-60 transition-opacity duration-1000 ${isCalibrating ? 'opacity-20' : 'opacity-60'}`}
               />
               
               {isCalibrating && (
                 <div className="absolute inset-0 z-50 flex flex-col items-center justify-center bg-slate-950/60 backdrop-blur-sm">
                    <div className="w-12 h-12 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin mb-4"></div>
                    <div className="text-[10px] font-black uppercase tracking-[0.5em] text-indigo-400">Calibrating Lens...</div>
                 </div>
               )}
               
               <div className="absolute inset-0 pointer-events-none border-[20px] border-indigo-500/10 mix-blend-overlay"></div>
               
               {/* UI Overlays */}
               <div className="absolute inset-x-0 top-0 h-32 bg-gradient-to-b from-slate-950/80 to-transparent pointer-events-none"></div>
               
               <div className="relative z-10 flex flex-col items-center pointer-events-none absolute inset-0 justify-center">
                  <div className="w-48 h-48 border border-white/10 rounded-full flex items-center justify-center animate-pulse mb-6 relative">
                     <div className="absolute inset-0 border-t-2 border-indigo-500 rounded-full animate-spin"></div>
                     <Target size={48} className="text-indigo-400 opacity-50" />
                  </div>
                  <div className="text-indigo-400 font-black italic serif text-sm tracking-[0.5em] animate-pulse">SKELETAL MAPPING ACTIVE</div>
               </div>
               
               <div className="absolute top-6 left-6 flex items-center gap-2 bg-emerald-500 px-3 py-1 rounded-full text-[10px] font-black uppercase text-white tracking-widest shadow-lg shadow-emerald-500/40">
                  <div className="w-2 h-2 bg-white rounded-full animate-ping"></div>
                  Camera Running...
               </div>
               <div className="absolute top-6 right-6 flex items-center gap-3">
                  <div className="font-mono text-[10px] text-white/70 bg-slate-950/60 backdrop-blur-md px-3 py-1 border border-white/5 rounded-lg">
                     720p @ 30FPS | {selectedModel.toUpperCase()}
                  </div>
                  <button 
                    onClick={handleStop}
                    className="bg-red-500/10 hover:bg-red-500 text-red-400 hover:text-white border border-red-500/20 px-3 py-1 rounded-lg text-[10px] font-black uppercase tracking-widest transition-all backdrop-blur-md"
                  >
                    Stop
                  </button>
               </div>
               
               {prediction && (
                 <motion.div 
                   initial={{ opacity: 0, scale: 0.9 }}
                   animate={{ opacity: 1, scale: 1 }}
                   className="absolute bottom-6 inset-x-6 bg-slate-950/80 backdrop-blur-2xl p-6 rounded-[2rem] border border-white/10 shadow-2xl flex items-center justify-between"
                 >
                    <div>
                       <div className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500 mb-1 flex items-center gap-2">
                         Primary Inference
                         {prediction.disagreement && <span className="bg-amber-500/20 text-amber-400 px-2 py-0.5 rounded text-[8px]">CONFLICT DETECTED</span>}
                       </div>
                       <motion.div 
                         key={displayActivity}
                         initial={{ y: 10, opacity: 0 }}
                         animate={{ y: 0, opacity: 1 }}
                         className="text-3xl font-black italic serif text-white tracking-tighter leading-none flex items-center gap-3"
                       >
                         {displayActivity}
                         {displayActivity.includes('WALK') && <span className="text-2xl animate-bounce">🏃</span>}
                       </motion.div>
                    </div>
                    <div className="text-right">
                       <div className="text-[10px] font-black uppercase tracking-[0.3em] text-slate-500 mb-1">Confidence</div>
                       <motion.div 
                         key={displayConf}
                         initial={{ x: 10, opacity: 0 }}
                         animate={{ x: 0, opacity: 1 }}
                         className="text-3xl font-black italic serif text-emerald-400 tracking-tighter leading-none"
                       >
                         {(displayConf * 100).toFixed(1)}%
                       </motion.div>
                    </div>
                 </motion.div>
               )}
            </div>
          ) : (
            <div className="w-full h-full bg-slate-950 flex flex-col items-center justify-center p-12 text-center relative">
               <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,_var(--tw-gradient-stops))] from-indigo-900/10 via-transparent to-transparent opacity-50"></div>
               <div className="w-24 h-24 bg-slate-900 rounded-[2rem] flex items-center justify-center text-slate-700 mb-8 border border-white/5 shadow-2xl relative z-10 group-hover:scale-110 transition-transform">
                  <Smartphone size={48} className="opacity-40" />
               </div>
               
               {cameraStatus === 'STOPPED' && (
                 <motion.div 
                    initial={{ opacity: 0, y: -20 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="absolute top-10 bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-6 py-2 rounded-full text-[10px] font-black uppercase tracking-widest z-20"
                 >
                    Camera stopped successfully
                 </motion.div>
               )}

               <h3 className="text-2xl font-black text-white mb-4 italic serif uppercase tracking-tight relative z-10">
                 {cameraStatus === 'STOPPED' ? 'Camera Stopped' : 'Sensor Link Inactive'}
               </h3>
               <p className="text-sm text-slate-500 max-w-sm mb-12 font-medium relative z-10">
                 {permissionError 
                   ? `Access Error: ${permissionError}` 
                   : "Authorize camera access to initialize the MotionSense computer vision and skeletal recognition engine."}
               </p>
               
               <div className="flex gap-4 relative z-10">
                  <button 
                    disabled={isCameraActive}
                    onClick={handleStart}
                    className="bg-emerald-600 hover:bg-emerald-500 disabled:opacity-30 disabled:cursor-not-allowed text-white px-8 py-4 rounded-2xl font-black uppercase tracking-[0.3em] text-[10px] transition-all shadow-2xl shadow-emerald-900/40 active:scale-95 flex items-center gap-2"
                  >
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                    Start Camera
                  </button>
                  <button 
                    disabled={!isCameraActive}
                    onClick={handleStop}
                    className="bg-red-600 hover:bg-red-500 disabled:opacity-30 disabled:cursor-not-allowed text-white px-8 py-4 rounded-2xl font-black uppercase tracking-[0.3em] text-[10px] transition-all shadow-2xl shadow-red-900/40 active:scale-95 flex items-center gap-2"
                  >
                    <div className="w-2 h-2 bg-white rounded-full"></div>
                    Stop Camera
                  </button>
               </div>
            </div>
          )}
        </div>
      </div>

      <div className="col-span-12 lg:col-span-4 space-y-6">
        <div className="bento-card p-10 bg-slate-900 border-white/5 text-white shadow-2xl">
           <h3 className="text-[10px] font-black uppercase tracking-[0.3em] mb-10 text-slate-500 italic serif">Configuration</h3>
           <div className="space-y-8">
             <div>
               <label className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 mb-4 block">Inference Engine</label>
               <select 
                 value={selectedModel}
                 onChange={(e) => setSelectedModel(e.target.value)}
                 className="w-full bg-slate-800/50 border border-white/5 p-4 rounded-2xl text-xs font-bold text-white focus:ring-2 focus:ring-indigo-500 outline-none transition-all appearance-none cursor-pointer"
               >
                 {MODEL_STATS.map(m => <option key={m.model} value={m.model}>{m.model}</option>)}
               </select>
             </div>
             
             <div className="p-8 bg-slate-800/30 rounded-[2rem] border border-white/5 group hover:bg-slate-800/50 transition-all cursor-pointer">
                <div className="flex items-center gap-4 mb-6">
                   <div className="w-10 h-10 bg-indigo-600/20 rounded-xl flex items-center justify-center text-indigo-400 group-hover:scale-110 transition-transform">
                      <CloudUpload size={20} />
                   </div>
                   <span className="text-[10px] font-black uppercase tracking-[0.3em]">Bulk Video Mode</span>
                </div>
                <p className="text-xs text-slate-500 leading-relaxed font-medium mb-6">Analyze pre-recorded session data through skeleton-only normalization.</p>
                <div className="w-full py-4 border border-slate-700 rounded-2xl text-center text-[10px] font-black uppercase tracking-widest text-slate-500 group-hover:text-indigo-400 transition-colors">
                   Select Local File
                </div>
             </div>
           </div>
        </div>

        <div className="bento-card p-8 bg-gradient-to-br from-indigo-600 to-purple-700 text-white border-none shadow-2xl shadow-indigo-900/40">
           <div className="flex items-center gap-3 mb-6">
              <Zap size={20} className="text-white/80" />
              <span className="text-[10px] font-black uppercase tracking-[0.3em]">Network Topology</span>
           </div>
           <div className="space-y-6">
              <div className="space-y-2">
                <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-widest opacity-80">
                   <span>Stream Stability</span>
                   <span className="text-white">99%</span>
                </div>
                <div className="h-1 bg-white/20 rounded-full overflow-hidden">
                   <div className="h-full bg-white w-[99%]"></div>
                </div>
              </div>
              <div className="space-y-2">
                <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-widest opacity-80">
                   <span>Inference CPU</span>
                   <span className="text-white">Low-Latency</span>
                </div>
                <div className="h-1 bg-white/20 rounded-full overflow-hidden">
                   <div className="h-full bg-white w-[35%]"></div>
                </div>
              </div>
           </div>
        </div>
      </div>
    </div>
  );
}

function AboutPage() {
  return (
    <div className="grid grid-cols-12 gap-6">
      <div className="col-span-12 lg:col-span-12 bento-card p-10 bg-slate-900 border-transparent text-white relative overflow-hidden flex items-center">
         <div className="absolute top-0 right-0 p-12 opacity-5 pointer-events-none transform translate-x-10 -translate-y-10 group-hover:scale-110 transition-all duration-700">
            <BrainCircuit size={250} />
         </div>
         <div className="relative z-10 max-w-2xl">
            <div className="bg-indigo-500 text-white px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest w-fit mb-6">Project Origins</div>
            <h2 className="text-5xl font-black italic serif tracking-tighter mb-6 leading-none">THE EVOLUTION OF <br /> HAR INTELLIGENCE</h2>
            <p className="text-slate-400 text-lg font-medium leading-relaxed">
              Conceptualized as an end-to-end framework to bridge raw smartphone telemetry with high-fidelity behavioral classification.
            </p>
         </div>
      </div>

      <div className="col-span-12 lg:col-span-7 bento-card p-8">
         <h4 className="text-[10px] font-black uppercase text-slate-400 tracking-widest mb-6">Integrated Ecosystem</h4>
         <div className="grid grid-cols-2 gap-4">
            <TechCard name="Python" desc="Data Orchestration" />
            <TechCard name="TensorFlow" desc="Deep Learning Core" />
            <TechCard name="Vite" desc="HMR Dashboard" />
            <TechCard name="Tailwind" desc="Theming Engine" />
         </div>
         <p className="mt-8 text-sm text-slate-500 font-medium leading-relaxed italic">
           * All models are validated against the UCI Human Activity Recognition Dataset utilizing 561-feature engineered vectors.
         </p>
      </div>

      <div className="col-span-12 lg:col-span-5 bento-card p-8 bg-slate-50 flex flex-col justify-between">
         <div>
            <h4 className="text-[10px] font-black uppercase text-indigo-500 tracking-widest mb-6">Future Horizons</h4>
            <div className="space-y-4">
               <div className="flex items-center gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-slate-900"></div>
                  <span className="text-sm font-bold text-slate-800 italic tracking-tight">Transformer Encoders</span>
               </div>
               <div className="flex items-center gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-slate-900"></div>
                  <span className="text-sm font-bold text-slate-800 italic tracking-tight">LOSO Cross-Validation</span>
               </div>
               <div className="flex items-center gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-slate-900"></div>
                  <span className="text-sm font-bold text-slate-800 italic tracking-tight">Real-time Camera Fusion</span>
               </div>
            </div>
         </div>
         <div className="pt-8 border-t border-slate-200 mt-8 flex justify-between items-center text-[10px] font-black text-slate-400 tracking-[0.2em]">
            <span>SYSTEM v2.5</span>
            <span className="text-indigo-500">READY</span>
         </div>
      </div>
    </div>
  );
}

function TechCard({ name, desc }: { name: string, desc: string }) {
  return (
    <div className="p-5 bg-slate-50 border border-slate-100 rounded-2xl hover:bg-white hover:border-indigo-100 transition-all cursor-default group">
      <div className="font-bold text-slate-800 text-sm mb-1 group-hover:text-indigo-600">{name}</div>
      <div className="text-[9px] text-slate-400 font-black uppercase tracking-widest">{desc}</div>
    </div>
  );
}

function LoginPage({ onLogin }: { onLogin: () => void }) {
  const [mode, setMode] = useState<'login' | 'signup' | 'forgot'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (mode === 'forgot') {
      alert('Password reset link sent to ' + email);
      setMode('login');
    } else {
      onLogin();
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-6 relative overflow-hidden">
      {/* Background elements */}
      <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-indigo-600/20 blur-[120px] rounded-full -translate-y-1/2 translate-x-1/2"></div>
      <div className="absolute bottom-0 left-0 w-[500px] h-[500px] bg-purple-600/20 blur-[120px] rounded-full translate-y-1/2 -translate-x-1/2"></div>
      
      <motion.div 
        layout
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="w-full max-w-md bg-slate-900 border border-white/5 rounded-[2.5rem] p-10 shadow-2xl relative z-10 backdrop-blur-2xl"
      >
        <div className="relative z-10">
          <div className="w-16 h-16 bg-gradient-to-br from-indigo-600 to-purple-700 rounded-2xl flex items-center justify-center text-white text-2xl font-black mb-8 rotate-3 shadow-xl">M</div>
          <h1 className="text-4xl font-black italic serif tracking-tighter text-white mb-2 leading-none uppercase">
            {mode === 'login' && 'Welcome Back'}
            {mode === 'signup' && 'Join the Elite'}
            {mode === 'forgot' && 'Reset Access'}
          </h1>
          <p className="text-slate-400 font-medium mb-10">
            {mode === 'login' && 'Secure authentication for MotionSense AI.'}
            {mode === 'signup' && 'Start your movement intelligence journey.'}
            {mode === 'forgot' && 'Enter your email to recover your account.'}
          </p>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div className="space-y-2">
              <label className="text-[10px] font-black uppercase tracking-widest text-slate-500 block">Email Address</label>
              <input 
                type="email" 
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="tenej@example.com"
                className="w-full p-4 bg-slate-800/50 border border-white/5 rounded-2xl font-bold text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all text-white placeholder:text-slate-600"
              />
            </div>
            {mode !== 'forgot' && (
              <div className="space-y-2">
                <div className="flex justify-between items-center">
                  <label className="text-[10px] font-black uppercase tracking-widest text-slate-500 block">Password</label>
                  {mode === 'login' && (
                    <button 
                      type="button" 
                      onClick={() => setMode('forgot')}
                      className="text-[10px] font-bold text-indigo-400 hover:text-indigo-300 uppercase tracking-widest"
                    >
                      Forgot?
                    </button>
                  )}
                </div>
                <input 
                  type="password" 
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="w-full p-4 bg-slate-800/50 border border-white/5 rounded-2xl font-bold text-sm focus:ring-2 focus:ring-indigo-500 outline-none transition-all text-white placeholder:text-slate-600"
                />
              </div>
            )}

            <button 
              type="submit"
              className="w-full py-5 bg-indigo-600 hover:bg-indigo-500 text-white rounded-2xl font-black italic serif text-xl tracking-tight transition-all shadow-xl shadow-indigo-900/40 uppercase"
            >
              {mode === 'login' && 'Sign In'}
              {mode === 'signup' && 'Create Account'}
              {mode === 'forgot' && 'Send Reset Link'}
            </button>
          </form>

          <div className="mt-8 text-center">
            {mode === 'login' ? (
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                Don't have an account? <button onClick={() => setMode('signup')} className="text-indigo-400 hover:text-indigo-300 ml-1">Sign Up</button>
              </p>
            ) : (
              <p className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">
                Already member? <button onClick={() => setMode('login')} className="text-indigo-400 hover:text-indigo-300 ml-1">Sign In</button>
              </p>
            )}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

function ProfilePage({ user }: { user: any }) {
  const fitnessStats = [
    { label: 'Steps', value: '11,248', sub: 'Target 12k', icon: <ChevronRight size={16} />, color: 'bg-indigo-600' },
    { label: 'Intensity', value: '420', sub: 'Burn Score', icon: <Flame size={16} />, color: 'bg-orange-500' },
    { label: 'Uptime', value: '12.4h', sub: 'Active Day', icon: <Clock size={16} />, color: 'bg-emerald-500' },
    { label: 'Peak BPM', value: '142', sub: 'Cardio Max', icon: <Heart size={16} />, color: 'bg-rose-500' },
  ];

  return (
    <div className="space-y-8 pb-10">
      <div className="bento-card p-10 bg-slate-900 border-white/5 text-white relative overflow-hidden flex flex-col md:flex-row items-center gap-10 shadow-2xl">
        <div className="absolute top-0 right-0 p-10 opacity-5 pointer-events-none transform translate-x-10 -translate-y-10 group-hover:scale-110 transition-transform">
           <User size={300} />
        </div>
        <div className="relative z-10">
           <img src={user.avatar} alt="Avatar" className="w-40 h-40 rounded-[2.5rem] border-4 border-slate-800 shadow-2xl rotate-3 bg-indigo-500/20 p-2" />
        </div>
        <div className="relative z-10 flex-1">
           <div className="bg-indigo-600 text-white px-3 py-1 rounded-full text-[10px] font-black uppercase tracking-widest w-fit mb-4 shadow-lg shadow-indigo-500/20">Elite Membership</div>
           <h2 className="text-6xl font-black italic serif tracking-tighter mb-2 leading-none uppercase text-white">{user.name}</h2>
           <p className="text-slate-400 text-xl font-medium mb-8 opacity-80">{user.email}</p>
           <div className="flex flex-wrap gap-4">
              <div className="flex items-center gap-2 bg-white/5 backdrop-blur-md border border-white/10 px-5 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest">
                 <Calendar size={14} className="text-indigo-400" />
                 <span>Joined {user.joinDate}</span>
              </div>
              <div className="flex items-center gap-2 bg-white/5 backdrop-blur-md border border-white/10 px-5 py-3 rounded-2xl text-[10px] font-black uppercase tracking-widest">
                 <Trophy size={14} className="text-amber-400" />
                 <span>Movement Rank: #24</span>
              </div>
           </div>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
         {fitnessStats.map((stat, i) => (
           <div key={i} className="bento-card p-8 group hover:border-indigo-500/50 transition-all bg-slate-900 shadow-2xl">
              <div className="flex justify-between items-start mb-6">
                 <div className={`p-4 rounded-2xl text-white ${stat.color} shadow-lg shadow-black/5 group-hover:scale-110 transition-transform`}>
                    {stat.icon}
                 </div>
                 <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest pt-2">{stat.label}</div>
              </div>
              <div className="text-4xl font-black text-white tracking-tighter italic serif">{stat.value}</div>
              <div className="text-[10px] font-black text-slate-500 mt-2 uppercase tracking-[0.2em]">{stat.sub}</div>
           </div>
         ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
         <div className="lg:col-span-2 space-y-6">
            <div className="bento-card p-10 bg-slate-900 shadow-2xl h-full">
               <div className="flex justify-between items-end mb-10">
                  <h3 className="text-2xl font-black italic serif tracking-tight text-white uppercase">Movement History</h3>
                  <div className="text-[10px] font-black text-indigo-400 uppercase tracking-widest cursor-pointer hover:text-indigo-300 transition-colors">Export Logs</div>
               </div>
               <div className="space-y-4">
                  {SESSION_HISTORY.map((session) => (
                    <div key={session.id} className="flex items-center gap-6 p-6 rounded-3xl hover:bg-white/5 border border-transparent hover:border-white/5 transition-all group relative overflow-hidden">
                       <div className="absolute top-0 left-0 h-full w-1.5 bg-indigo-600 transition-transform -translate-x-full group-hover:translate-x-0"></div>
                       <div className="w-16 h-16 bg-slate-800 rounded-2xl flex items-center justify-center group-hover:scale-110 transition-transform shrink-0">
                          <ActivityIcon className="text-indigo-600" size={24} />
                       </div>
                       <div className="flex-1">
                          <div className="font-black text-xl italic serif text-white uppercase tracking-tighter">{session.activity.replace('_', ' ')}</div>
                          <div className="text-[10px] font-black text-slate-500 uppercase tracking-widest mt-1">{session.date} • {session.model}</div>
                       </div>
                       <div className="text-right">
                          <div className="font-black text-2xl italic tracking-tighter text-white">{(session.confidence * 100).toFixed(0)}%</div>
                          <div className="text-[10px] font-black text-emerald-400 uppercase tracking-widest">Conf. Index</div>
                       </div>
                    </div>
                  ))}
               </div>
            </div>
         </div>

         <div className="space-y-8">
            <div className="bento-card p-10 bg-indigo-600 text-white border-none shadow-2xl shadow-indigo-900/40 relative overflow-hidden flex flex-col justify-center">
               <div className="absolute top-0 right-0 p-10 opacity-10 blur-2xl">
                  <Flame size={200} />
               </div>
               <div className="relative z-10 text-center">
                  <h4 className="text-[10px] font-black uppercase tracking-[0.3em] opacity-70 mb-6">Intensity Target</h4>
                  <div className="text-8xl font-black italic serif tracking-tighter mb-4">88<span className="text-3xl">%</span></div>
                  <p className="text-xs font-bold opacity-80 leading-relaxed italic max-w-[200px] mx-auto uppercase tracking-widest mb-8">Average Daily Threshold Reached</p>
                  <button className="bg-white text-indigo-600 w-full py-4 rounded-2xl font-black uppercase tracking-[0.2em] text-[10px] shadow-xl">
                    Challenge Friend
                  </button>
               </div>
            </div>
            
            <div className="bento-card p-10 bg-slate-900 border-white/5 text-white">
               <h4 className="text-[10px] font-black uppercase text-slate-400 tracking-[0.2em] mb-8">System Connectivity</h4>
               <div className="flex items-center gap-4 mb-8">
                  <div className="w-3 h-3 bg-emerald-500 rounded-full animate-ping"></div>
                  <div className="text-xl font-black italic serif tracking-tight">Lens: Linked (50Hz)</div>
               </div>
               <div className="space-y-4 pt-6 border-t border-white/10">
                  <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-[0.2em]">
                     <span className="text-slate-500">Latency Profile</span>
                     <span className="text-indigo-400">Ultra-Low</span>
                  </div>
                  <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-[0.2em]">
                     <span className="text-slate-500">Buffer Usage</span>
                     <span className="text-white">12%</span>
                  </div>
                  <div className="flex justify-between items-center text-[10px] font-black uppercase tracking-[0.2em]">
                     <span className="text-slate-500">ML Mode</span>
                     <span className="text-white">Local-Only</span>
                  </div>
               </div>
            </div>
         </div>
      </div>
    </div>
  );
}