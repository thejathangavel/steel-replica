'use client'

import { useState, useEffect, FormEvent } from 'react'
import { useRouter } from 'next/navigation'
import { supabase } from '../../lib/supabase'
import { apiFetch } from '../../lib/api'

interface Project {
  id: string
  name: string
  project_number: string | null
  design_standard: string
  unit_system: string
  location: string | null
  status: string
  created_at: string
}

interface NewProjectForm {
  name: string
  project_number: string
  design_standard: string
  unit_system: string
  location: string
}

const STANDARDS = ['AISC', 'IS 800', 'Eurocode 3', 'BS 5950', 'AS 4100']
const UNITS     = ['imperial', 'metric']

export default function DashboardPage() {
  const router = useRouter()
  const [projects, setProjects]   = useState<Project[]>([])
  const [loading, setLoading]     = useState(true)
  const [userEmail, setUserEmail] = useState('')
  const [showModal, setShowModal] = useState(false)
  const [creating, setCreating]   = useState(false)
  const [formError, setFormError] = useState<string | null>(null)
  const [form, setForm] = useState<NewProjectForm>({
    name: '', project_number: '', design_standard: 'AISC',
    unit_system: 'imperial', location: '',
  })

  // Auth guard — redirect to login if no session
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (!session) { router.push('/login'); return }
      setUserEmail(session.user.email ?? '')
      loadProjects()
    })
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  async function loadProjects() {
    setLoading(true)
    try {
      const data = await apiFetch('/projects')
      setProjects(data)
    } catch {
      setProjects([])
    } finally {
      setLoading(false)
    }
  }

  async function handleSignOut() {
    await supabase.auth.signOut()
    router.push('/login')
  }

  async function handleCreateProject(e: FormEvent) {
    e.preventDefault()
    setFormError(null)
    setCreating(true)
    try {
      const created = await apiFetch('/projects', {
        method: 'POST',
        body: JSON.stringify({
          name:            form.name,
          project_number:  form.project_number || undefined,
          design_standard: form.design_standard,
          unit_system:     form.unit_system,
          location:        form.location || undefined,
        }),
      })
      setProjects(prev => [created, ...prev])
      setShowModal(false)
      setForm({ name:'', project_number:'', design_standard:'AISC', unit_system:'imperial', location:'' })
    } catch (err: unknown) {
      setFormError(err instanceof Error ? err.message : 'Failed to create project')
    } finally {
      setCreating(false)
    }
  }

  function openProject(project: Project) {
    // Week 3: the project_id query param will be read by page.tsx to
    // associate PDF uploads with this project via POST /projects/{id}/upload.
    // For now it is passed in the URL but not yet consumed by the viewer.
    router.push(`/?project_id=${project.id}`)
  }

  function formatDate(iso: string) {
    return new Date(iso).toLocaleDateString('en-US', { month: 'short', day: 'numeric', year: 'numeric' })
  }

  function statusColor(status: string) {
    if (status === 'complete') return '#34D399'
    if (status === 'in_progress') return '#FBBF24'
    return '#64748B'
  }

  return (
    <div style={{ minHeight:'100vh', backgroundColor:'#0F172A', fontFamily:"'Inter', system-ui, sans-serif" }}>
      {/* Background grid */}
      <div style={{ position:'fixed', inset:0, pointerEvents:'none', zIndex:0,
        backgroundImage:'linear-gradient(rgba(59,130,246,0.03) 1px,transparent 1px),linear-gradient(90deg,rgba(59,130,246,0.03) 1px,transparent 1px)',
        backgroundSize:'40px 40px' }} />

      {/* Header */}
      <header style={{ position:'sticky', top:0, zIndex:50, backgroundColor:'rgba(9,14,27,0.9)',
          backdropFilter:'blur(12px)', borderBottom:'1px solid rgba(59,130,246,0.12)',
          padding:'0 32px', height:'60px', display:'flex', alignItems:'center', justifyContent:'space-between' }}>
        <div style={{ display:'flex', alignItems:'center', gap:'10px' }}>
          <div style={{ width:'32px', height:'32px', backgroundColor:'rgba(59,130,246,0.12)',
              border:'1px solid rgba(59,130,246,0.3)', borderRadius:'8px',
              display:'flex', alignItems:'center', justifyContent:'center' }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
              <path d="M4 4h16v4H4zM4 10h10v4H4zM4 16h7v4H4z" fill="#3B82F6" />
            </svg>
          </div>
          <span style={{ fontSize:'16px', fontWeight:700, color:'#F1F5F9', letterSpacing:'-0.3px' }}>SteelGenie</span>
          <span style={{ marginLeft:'8px', padding:'2px 8px', backgroundColor:'rgba(59,130,246,0.1)',
              border:'1px solid rgba(59,130,246,0.2)', borderRadius:'4px',
              fontSize:'10px', fontWeight:600, color:'#60A5FA', letterSpacing:'0.8px' }}>
            DASHBOARD
          </span>
        </div>
        <div style={{ display:'flex', alignItems:'center', gap:'16px' }}>
          <span style={{ fontSize:'13px', color:'#64748B' }}>{userEmail}</span>
          <button onClick={handleSignOut}
            style={{ padding:'6px 14px', backgroundColor:'transparent', border:'1px solid rgba(100,116,139,0.3)',
              borderRadius:'7px', color:'#94A3B8', fontSize:'12px', fontWeight:600, cursor:'pointer',
              fontFamily:'inherit', transition:'all 0.2s' }}>
            Sign Out
          </button>
        </div>
      </header>

      {/* Main content */}
      <main style={{ position:'relative', zIndex:10, maxWidth:'1200px', margin:'0 auto', padding:'40px 32px' }}>
        {/* Page title row */}
        <div style={{ display:'flex', alignItems:'flex-start', justifyContent:'space-between',
            marginBottom:'36px', flexWrap:'wrap', gap:'16px' }}>
          <div>
            <h1 style={{ margin:'0 0 6px', fontSize:'28px', fontWeight:700, color:'#F1F5F9', letterSpacing:'-0.5px' }}>
              Projects
            </h1>
            <p style={{ margin:0, fontSize:'14px', color:'#64748B' }}>
              Manage your structural drawing projects
            </p>
          </div>
          <button onClick={() => setShowModal(true)}
            style={{ display:'flex', alignItems:'center', gap:'8px', padding:'10px 20px',
              backgroundColor:'#3B82F6', border:'none', borderRadius:'9px', color:'#fff',
              fontSize:'14px', fontWeight:600, cursor:'pointer', fontFamily:'inherit',
              boxShadow:'0 4px 14px rgba(59,130,246,0.35)', transition:'all 0.2s' }}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
              <path d="M12 5v14M5 12h14" stroke="#fff" strokeWidth="2.5" strokeLinecap="round"/>
            </svg>
            New Project
          </button>
        </div>

        {/* Stats bar */}
        <div style={{ display:'flex', gap:'16px', marginBottom:'32px', flexWrap:'wrap' }}>
          {[
            { label:'Total Projects', value: projects.length },
            { label:'In Progress',    value: projects.filter(p => p.status === 'in_progress').length },
            { label:'Completed',      value: projects.filter(p => p.status === 'complete').length },
          ].map(stat => (
            <div key={stat.label} style={{ flex:'1 1 140px', backgroundColor:'rgba(30,41,59,0.5)',
                border:'1px solid rgba(59,130,246,0.1)', borderRadius:'10px', padding:'16px 20px' }}>
              <div style={{ fontSize:'24px', fontWeight:700, color:'#F1F5F9' }}>{stat.value}</div>
              <div style={{ fontSize:'12px', color:'#64748B', marginTop:'2px' }}>{stat.label}</div>
            </div>
          ))}
        </div>

        {/* Project grid */}
        {loading ? (
          <div style={{ display:'flex', justifyContent:'center', padding:'60px', color:'#64748B', fontSize:'14px' }}>
            Loading projects…
          </div>
        ) : projects.length === 0 ? (
          <div style={{ display:'flex', flexDirection:'column', alignItems:'center', justifyContent:'center',
              padding:'80px 20px', textAlign:'center' }}>
            <div style={{ width:'60px', height:'60px', backgroundColor:'rgba(59,130,246,0.08)',
                border:'1px solid rgba(59,130,246,0.2)', borderRadius:'14px',
                display:'flex', alignItems:'center', justifyContent:'center', marginBottom:'20px' }}>
              <svg width="28" height="28" viewBox="0 0 24 24" fill="none">
                <path d="M4 4h16v4H4zM4 10h10v4H4zM4 16h7v4H4z" fill="#3B82F6" opacity="0.6"/>
              </svg>
            </div>
            <h2 style={{ margin:'0 0 8px', fontSize:'18px', fontWeight:600, color:'#94A3B8' }}>
              No projects yet
            </h2>
            <p style={{ margin:'0 0 24px', fontSize:'14px', color:'#475569' }}>
              Create your first project to start analysing structural drawings.
            </p>
            <button onClick={() => setShowModal(true)}
              style={{ padding:'10px 24px', backgroundColor:'#3B82F6', border:'none', borderRadius:'9px',
                color:'#fff', fontSize:'14px', fontWeight:600, cursor:'pointer', fontFamily:'inherit',
                boxShadow:'0 4px 14px rgba(59,130,246,0.35)' }}>
              Create Project
            </button>
          </div>
        ) : (
          <div style={{ display:'grid', gridTemplateColumns:'repeat(auto-fill, minmax(320px, 1fr))', gap:'20px' }}>
            {projects.map(project => (
              <div key={project.id}
                style={{ backgroundColor:'rgba(15,23,42,0.8)', border:'1px solid rgba(59,130,246,0.12)',
                  borderRadius:'12px', padding:'24px', transition:'all 0.2s', cursor:'default',
                  boxShadow:'0 4px 20px rgba(0,0,0,0.2)' }}
                onMouseEnter={e => { (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(59,130,246,0.35)'; (e.currentTarget as HTMLDivElement).style.boxShadow = '0 8px 30px rgba(0,0,0,0.3)' }}
                onMouseLeave={e => { (e.currentTarget as HTMLDivElement).style.borderColor = 'rgba(59,130,246,0.12)'; (e.currentTarget as HTMLDivElement).style.boxShadow = '0 4px 20px rgba(0,0,0,0.2)' }}>

                {/* Card header */}
                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'flex-start', marginBottom:'14px' }}>
                  <div style={{ width:'38px', height:'38px', backgroundColor:'rgba(59,130,246,0.1)',
                      border:'1px solid rgba(59,130,246,0.2)', borderRadius:'9px',
                      display:'flex', alignItems:'center', justifyContent:'center', flexShrink:0 }}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none">
                      <path d="M4 4h16v4H4zM4 10h10v4H4zM4 16h7v4H4z" fill="#3B82F6" opacity="0.8"/>
                    </svg>
                  </div>
                  <span style={{ padding:'3px 9px', borderRadius:'5px', fontSize:'11px', fontWeight:600,
                      backgroundColor: project.status==='complete' ? 'rgba(52,211,153,0.1)' : 'rgba(251,191,36,0.1)',
                      border: `1px solid ${project.status==='complete' ? 'rgba(52,211,153,0.25)' : 'rgba(251,191,36,0.25)'}`,
                      color: statusColor(project.status), textTransform:'uppercase', letterSpacing:'0.5px' }}>
                    {project.status.replace('_',' ')}
                  </span>
                </div>

                {/* Project name */}
                <h2 style={{ margin:'0 0 4px', fontSize:'17px', fontWeight:700, color:'#F1F5F9',
                    letterSpacing:'-0.3px', lineHeight:1.3 }}>
                  {project.name}
                </h2>

                {project.project_number && (
                  <p style={{ margin:'0 0 14px', fontSize:'12px', color:'#60A5FA', fontWeight:500 }}>
                    #{project.project_number}
                  </p>
                )}

                {/* Meta pills */}
                <div style={{ display:'flex', flexWrap:'wrap', gap:'6px', marginBottom:'20px' }}>
                  {[
                    { label: project.design_standard },
                    { label: project.unit_system },
                    ...(project.location ? [{ label: project.location }] : []),
                  ].map((pill, i) => (
                    <span key={i} style={{ padding:'3px 9px', backgroundColor:'rgba(30,41,59,0.8)',
                        border:'1px solid rgba(100,116,139,0.15)', borderRadius:'5px',
                        fontSize:'11px', color:'#94A3B8', fontWeight:500 }}>
                      {pill.label}
                    </span>
                  ))}
                </div>

                <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center' }}>
                  <span style={{ fontSize:'12px', color:'#475569' }}>{formatDate(project.created_at)}</span>
                  <button onClick={() => openProject(project)}
                    style={{ display:'flex', alignItems:'center', gap:'6px', padding:'7px 16px',
                      backgroundColor:'rgba(59,130,246,0.12)', border:'1px solid rgba(59,130,246,0.25)',
                      borderRadius:'7px', color:'#60A5FA', fontSize:'13px', fontWeight:600,
                      cursor:'pointer', fontFamily:'inherit', transition:'all 0.2s' }}
                    onMouseEnter={e => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = '#3B82F6'; (e.currentTarget as HTMLButtonElement).style.color = '#fff' }}
                    onMouseLeave={e => { (e.currentTarget as HTMLButtonElement).style.backgroundColor = 'rgba(59,130,246,0.12)'; (e.currentTarget as HTMLButtonElement).style.color = '#60A5FA' }}>
                    Open
                    <svg width="12" height="12" viewBox="0 0 24 24" fill="none">
                      <path d="M5 12h14M12 5l7 7-7 7" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                    </svg>
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>

      {/* New Project Modal */}
      {showModal && (
        <div style={{ position:'fixed', inset:0, zIndex:100, display:'flex', alignItems:'center',
            justifyContent:'center', backgroundColor:'rgba(0,0,0,0.6)', backdropFilter:'blur(4px)',
            padding:'20px' }}
          onClick={e => { if (e.target === e.currentTarget) setShowModal(false) }}>
          <div style={{ width:'100%', maxWidth:'480px', backgroundColor:'#0F172A',
              border:'1px solid rgba(59,130,246,0.25)', borderRadius:'16px', padding:'36px',
              boxShadow:'0 25px 60px rgba(0,0,0,0.6)' }}>
            <div style={{ display:'flex', justifyContent:'space-between', alignItems:'center', marginBottom:'28px' }}>
              <h2 style={{ margin:0, fontSize:'20px', fontWeight:700, color:'#F1F5F9' }}>New Project</h2>
              <button onClick={() => setShowModal(false)}
                style={{ background:'none', border:'none', color:'#64748B', cursor:'pointer',
                  fontSize:'22px', lineHeight:1, padding:'0 4px', fontFamily:'inherit' }}>
                ×
              </button>
            </div>

            <form onSubmit={handleCreateProject} style={{ display:'flex', flexDirection:'column', gap:'16px' }}>
              {/* Project name */}
              <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
                <label style={{ fontSize:'12px', fontWeight:600, color:'#94A3B8', textTransform:'uppercase', letterSpacing:'0.6px' }}>
                  Project Name *
                </label>
                <input required value={form.name} onChange={e => setForm(f => ({...f, name: e.target.value}))}
                  placeholder="e.g. West Tower Framing"
                  style={{ padding:'10px 13px', backgroundColor:'rgba(30,41,59,0.8)',
                    border:'1px solid rgba(59,130,246,0.2)', borderRadius:'8px', color:'#F1F5F9',
                    fontSize:'14px', outline:'none', fontFamily:'inherit' }} />
              </div>

              {/* Project number */}
              <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
                <label style={{ fontSize:'12px', fontWeight:600, color:'#94A3B8', textTransform:'uppercase', letterSpacing:'0.6px' }}>
                  Project Number
                </label>
                <input value={form.project_number} onChange={e => setForm(f => ({...f, project_number: e.target.value}))}
                  placeholder="e.g. SG-2024-001"
                  style={{ padding:'10px 13px', backgroundColor:'rgba(30,41,59,0.8)',
                    border:'1px solid rgba(59,130,246,0.2)', borderRadius:'8px', color:'#F1F5F9',
                    fontSize:'14px', outline:'none', fontFamily:'inherit' }} />
              </div>

              {/* Design standard + unit system */}
              <div style={{ display:'grid', gridTemplateColumns:'1fr 1fr', gap:'12px' }}>
                <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
                  <label style={{ fontSize:'12px', fontWeight:600, color:'#94A3B8', textTransform:'uppercase', letterSpacing:'0.6px' }}>
                    Standard
                  </label>
                  <select value={form.design_standard} onChange={e => setForm(f => ({...f, design_standard: e.target.value}))}
                    style={{ padding:'10px 13px', backgroundColor:'rgba(30,41,59,0.8)',
                      border:'1px solid rgba(59,130,246,0.2)', borderRadius:'8px', color:'#F1F5F9',
                      fontSize:'14px', outline:'none', fontFamily:'inherit' }}>
                    {STANDARDS.map(s => <option key={s} value={s}>{s}</option>)}
                  </select>
                </div>
                <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
                  <label style={{ fontSize:'12px', fontWeight:600, color:'#94A3B8', textTransform:'uppercase', letterSpacing:'0.6px' }}>
                    Units
                  </label>
                  <select value={form.unit_system} onChange={e => setForm(f => ({...f, unit_system: e.target.value}))}
                    style={{ padding:'10px 13px', backgroundColor:'rgba(30,41,59,0.8)',
                      border:'1px solid rgba(59,130,246,0.2)', borderRadius:'8px', color:'#F1F5F9',
                      fontSize:'14px', outline:'none', fontFamily:'inherit' }}>
                    {UNITS.map(u => <option key={u} value={u}>{u}</option>)}
                  </select>
                </div>
              </div>

              {/* Location */}
              <div style={{ display:'flex', flexDirection:'column', gap:'6px' }}>
                <label style={{ fontSize:'12px', fontWeight:600, color:'#94A3B8', textTransform:'uppercase', letterSpacing:'0.6px' }}>
                  Location
                </label>
                <input value={form.location} onChange={e => setForm(f => ({...f, location: e.target.value}))}
                  placeholder="e.g. Chicago, IL"
                  style={{ padding:'10px 13px', backgroundColor:'rgba(30,41,59,0.8)',
                    border:'1px solid rgba(59,130,246,0.2)', borderRadius:'8px', color:'#F1F5F9',
                    fontSize:'14px', outline:'none', fontFamily:'inherit' }} />
              </div>

              {formError && (
                <div style={{ padding:'10px 14px', backgroundColor:'rgba(239,68,68,0.08)',
                    border:'1px solid rgba(239,68,68,0.2)', borderRadius:'8px',
                    color:'#F87171', fontSize:'13px' }}>
                  ⚠ {formError}
                </div>
              )}

              <div style={{ display:'flex', gap:'10px', marginTop:'4px' }}>
                <button type="button" onClick={() => setShowModal(false)}
                  style={{ flex:1, padding:'11px', backgroundColor:'transparent',
                    border:'1px solid rgba(100,116,139,0.25)', borderRadius:'8px',
                    color:'#94A3B8', fontSize:'14px', fontWeight:600, cursor:'pointer', fontFamily:'inherit' }}>
                  Cancel
                </button>
                <button type="submit" disabled={creating}
                  style={{ flex:2, padding:'11px', backgroundColor: creating ? '#1E40AF' : '#3B82F6',
                    border:'none', borderRadius:'8px', color:'#fff', fontSize:'14px', fontWeight:600,
                    cursor: creating ? 'not-allowed' : 'pointer', fontFamily:'inherit',
                    boxShadow: creating ? 'none' : '0 4px 14px rgba(59,130,246,0.35)' }}>
                  {creating ? 'Creating…' : 'Create Project'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  )
}
