'use client'

import { useState, FormEvent } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '../../../lib/supabase'

type Tab = 'login' | 'signup'

export default function LoginPage() {
  const router = useRouter()
  const [tab, setTab] = useState<Tab>('login')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [success, setSuccess] = useState<string | null>(null)

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setError(null)
    setSuccess(null)
    setLoading(true)
    try {
      if (tab === 'login') {
        const { error: err } = await supabase.auth.signInWithPassword({ email, password })
        if (err) throw err
        router.push('/dashboard')
      } else {
        const { error: err } = await supabase.auth.signUp({ email, password })
        if (err) throw err
        setSuccess('Account created! Signing you in…')
        const { error: err2 } = await supabase.auth.signInWithPassword({ email, password })
        if (err2) throw err2
        router.push('/dashboard')
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div style={{ minHeight:'100vh', backgroundColor:'#0F172A', display:'flex', alignItems:'center',
        justifyContent:'center', position:'relative', overflow:'hidden',
        fontFamily:"'Inter', system-ui, sans-serif" }}>
      {/* Grid */}
      <div style={{ position:'absolute', inset:0, pointerEvents:'none',
        backgroundImage:'linear-gradient(rgba(59,130,246,0.04) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,0.04) 1px,transparent 1px)',
        backgroundSize:'40px 40px' }} />
      {/* Glow blobs */}
      <div style={{ position:'absolute', top:'-15%', left:'-10%', width:'600px', height:'600px',
        borderRadius:'50%', background:'radial-gradient(circle,rgba(59,130,246,0.14) 0%,transparent 70%)', pointerEvents:'none' }} />
      <div style={{ position:'absolute', bottom:'-15%', right:'-10%', width:'600px', height:'600px',
        borderRadius:'50%', background:'radial-gradient(circle,rgba(99,102,241,0.10) 0%,transparent 70%)', pointerEvents:'none' }} />

      <div style={{ position:'relative', zIndex:10, width:'100%', maxWidth:'420px', margin:'0 16px',
          backgroundColor:'rgba(15,23,42,0.85)', border:'1px solid rgba(59,130,246,0.2)',
          borderRadius:'16px', padding:'40px 36px', backdropFilter:'blur(20px)',
          boxShadow:'0 25px 60px rgba(0,0,0,0.5)' }}>
        {/* Logo */}
        <div style={{ display:'flex', alignItems:'center', gap:'10px', marginBottom:'28px' }}>
          <div style={{ width:'40px', height:'40px', backgroundColor:'rgba(59,130,246,0.12)',
              border:'1px solid rgba(59,130,246,0.3)', borderRadius:'10px',
              display:'flex', alignItems:'center', justifyContent:'center' }}>
            <svg width="22" height="22" viewBox="0 0 24 24" fill="none">
              <path d="M4 4h16v4H4zM4 10h10v4H4zM4 16h7v4H4z" fill="#3B82F6" />
            </svg>
          </div>
          <span style={{ fontSize:'18px', fontWeight:700, color:'#F1F5F9', letterSpacing:'-0.3px' }}>SteelGenie</span>
        </div>

        <h1 style={{ margin:'0 0 6px', fontSize:'26px', fontWeight:700, color:'#F1F5F9', letterSpacing:'-0.5px' }}>
          {tab === 'login' ? 'Welcome back' : 'Create account'}
        </h1>
        <p style={{ margin:'0 0 28px', fontSize:'14px', color:'#64748B' }}>
          {tab === 'login' ? 'Sign in to your structural workspace' : 'Start your free structural analysis workspace'}
        </p>

        {/* Tabs */}
        <div style={{ display:'flex', gap:'4px', backgroundColor:'rgba(30,41,59,0.8)',
            border:'1px solid rgba(59,130,246,0.1)', borderRadius:'10px', padding:'4px', marginBottom:'28px' }}>
          {(['login','signup'] as Tab[]).map(t => (
            <button key={t} onClick={() => { setTab(t); setError(null); setSuccess(null) }}
              style={{ flex:1, padding:'8px 0', border:'none', borderRadius:'7px', fontSize:'13px',
                fontWeight:600, cursor:'pointer', transition:'all 0.2s', fontFamily:'inherit',
                backgroundColor: tab===t ? '#3B82F6' : 'transparent',
                color: tab===t ? '#fff' : '#64748B',
                boxShadow: tab===t ? '0 2px 8px rgba(59,130,246,0.35)' : 'none' }}>
              {t === 'login' ? 'Sign In' : 'Sign Up'}
            </button>
          ))}
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} style={{ display:'flex', flexDirection:'column', gap:'18px' }}>
          {['email','password'].map(field => (
            <div key={field} style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
              <label htmlFor={field} style={{ fontSize:'12px', fontWeight:600, color:'#94A3B8',
                  textTransform:'uppercase', letterSpacing:'0.6px' }}>
                {field === 'email' ? 'Email address' : 'Password'}
              </label>
              <input id={field} type={field} required minLength={field==='password' ? 6 : undefined}
                value={field==='email' ? email : password}
                onChange={e => field==='email' ? setEmail(e.target.value) : setPassword(e.target.value)}
                placeholder={field==='email' ? 'engineer@firm.com' : '••••••••'}
                style={{ padding:'11px 14px', backgroundColor:'rgba(30,41,59,0.8)',
                  border:'1px solid rgba(59,130,246,0.2)', borderRadius:'8px', color:'#F1F5F9',
                  fontSize:'14px', outline:'none', fontFamily:'inherit' }} />
            </div>
          ))}

          {error && (
            <div style={{ display:'flex', alignItems:'center', gap:'8px', padding:'10px 14px',
                backgroundColor:'rgba(239,68,68,0.08)', border:'1px solid rgba(239,68,68,0.2)',
                borderRadius:'8px', color:'#F87171', fontSize:'13px' }}>
              ⚠ {error}
            </div>
          )}
          {success && (
            <div style={{ display:'flex', alignItems:'center', gap:'8px', padding:'10px 14px',
                backgroundColor:'rgba(52,211,153,0.08)', border:'1px solid rgba(52,211,153,0.2)',
                borderRadius:'8px', color:'#34D399', fontSize:'13px' }}>
              ✓ {success}
            </div>
          )}

          <button type="submit" disabled={loading}
            style={{ marginTop:'4px', padding:'13px', backgroundColor: loading ? '#1E40AF' : '#3B82F6',
              border:'none', borderRadius:'9px', color:'#fff', fontSize:'15px', fontWeight:600,
              cursor: loading ? 'not-allowed' : 'pointer', fontFamily:'inherit',
              boxShadow: loading ? 'none' : '0 4px 14px rgba(59,130,246,0.35)', transition:'all 0.2s' }}>
            {loading ? 'Please wait…' : tab==='login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>

        <p style={{ marginTop:'24px', textAlign:'center', fontSize:'13px', color:'#475569' }}>
          {tab==='login' ? "Don't have an account? " : 'Already have an account? '}
          <button onClick={() => { setTab(tab==='login' ? 'signup' : 'login'); setError(null) }}
            style={{ background:'none', border:'none', color:'#60A5FA', fontSize:'13px',
              fontWeight:600, cursor:'pointer', padding:0, fontFamily:'inherit' }}>
            {tab==='login' ? 'Sign up free' : 'Sign in'}
          </button>
        </p>
      </div>
    </div>
  )
}
