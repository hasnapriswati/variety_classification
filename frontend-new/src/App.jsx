import { Link, NavLink, Route, Routes, Navigate, useNavigate, useLocation } from 'react-router-dom'
import { useState, useEffect, useRef } from 'react'
import { supabase } from './lib/supabaseClient'
import logoModern from './assets/logo-modern.svg'
import Login from './pages/Login.jsx'
import SignUp from './pages/SignUp.jsx'
import Upload from './pages/Upload.jsx'
import LiveDetect from './pages/LiveDetect.jsx'
import Results from './pages/Results.jsx'
import History from './pages/History.jsx'
import Reports from './pages/Reports.jsx'
import Dashboard from './pages/Dashboard.jsx'
import UserManagement from './pages/UserManagement.jsx'
import AuthCallback from './pages/AuthCallback.jsx'
import ResetPassword from './pages/ResetPassword.jsx'
import ForgotPassword from './pages/ForgotPassword.jsx'

const BRAND = 'ChiliVar'

function Navbar({ authed, onLogout, userName, userEmail, userRole }) {
  const [open, setOpen] = useState(false)
  // Guard untuk mencegah "ghost click" tepat setelah transisi login
  const [ready, setReady] = useState(false)
  const menuRef = useRef(null)
  const triggerRef = useRef(null)

  useEffect(() => {
    function onDocClick(e) {
      if (!open) return
      const t = e.target
      if (menuRef.current && !menuRef.current.contains(t) && triggerRef.current && !triggerRef.current.contains(t)) {
        setOpen(false)
      }
    }
    function onKey(e) {
      if (e.key === 'Escape') setOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('mousedown', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [open])
  
  // Aktifkan guard setiap kali status autentikasi berubah (mis. setelah login)
  useEffect(() => {
    setOpen(false)
    setReady(false)
    const t = setTimeout(() => setReady(true), 1200)
    return () => clearTimeout(t)
  }, [authed])

  return (
    <header className="app-header">
      <nav className="container nav">
        <div className="brand"><img src={logoModern} alt={BRAND} className="brand-logo" /><span className="brand-name">{BRAND}</span></div>
        {authed && (
          <div className="menu-center">
            <div className="top-tabs" role="tablist" aria-label="Navigasi utama">
              <NavLink to="/dashboard" className={({ isActive }) => `top-tab${isActive ? ' active' : ''}`} role="tab">📊 Dashboard</NavLink>
              <NavLink to="/upload" className={({ isActive }) => `top-tab${isActive ? ' active' : ''}`} role="tab">📤 Unggah</NavLink>
              <NavLink to="/live" className={({ isActive }) => `top-tab${isActive ? ' active' : ''}`} role="tab">📷 Deteksi</NavLink>
              <NavLink to="/history" className={({ isActive }) => `top-tab${isActive ? ' active' : ''}`} role="tab">🕘 Riwayat</NavLink>
              <NavLink to="/reports" className={({ isActive }) => `top-tab${isActive ? ' active' : ''}`} role="tab">📄 Unduh</NavLink>
              {/* Tautan admin akan ditambahkan di Sidebar; top tabs tetap ringkas */}
            </div>
          </div>
        )}
        <div className="links">
          {!authed && <Link to="/login">Login</Link>}
          {!authed && <Link to="/signup" style={{ marginLeft: 16 }}>Sign Up</Link>}
          {authed && (
            <div className="profile-wrapper" style={{ marginLeft: 12 }}>
              <button
                ref={triggerRef}
                className="profile-chip profile-trigger"
                title={userName || 'Pengguna'}
                aria-haspopup="menu"
                aria-expanded={open ? 'true' : 'false'}
                style={{ pointerEvents: ready ? 'auto' : 'none' }}
                onClickCapture={(e) => {
                  if (!ready) {
                    e.preventDefault()
                    e.stopPropagation()
                  }
                }}
                onMouseDown={(e) => {
                  if (!ready) {
                    e.preventDefault()
                    e.stopPropagation()
                  }
                }}
                onKeyDown={(e) => {
                  if (!ready && (e.key === 'Enter' || e.key === ' ')) {
                    e.preventDefault()
                    e.stopPropagation()
                  }
                }}
                onClick={(e) => {
                  // Abaikan klik pertama segera setelah login/navigasi untuk hindari popup auto-terbuka
                  if (!ready) return
                  setOpen(v => !v)
                }}
              >
                <div className="avatar" aria-hidden="true">{(userName || 'U').charAt(0).toUpperCase()}</div>
                <span className="name">{userName || 'Pengguna'}</span>
                <span className="chev" aria-hidden>▾</span>
              </button>
              {open && ready && (
                <div ref={menuRef} className="profile-menu" role="menu">
                  <div className="menu-header">
                    <div className="avatar big">{(userName || 'U').charAt(0).toUpperCase()}</div>
                    <div className="user-meta">
                      <div className="user-name">{userName || 'Pengguna'}</div>
                      <div className="user-sub">{userEmail || 'Akun'}</div>
                    </div>
                  </div>
                  
                  {String(userRole).trim().toLowerCase() === 'admin' && (
                    <Link to="/admin/users" className="menu-item" role="menuitem">Kelola Pengguna</Link>
                  )}
                  
                  <button className="menu-item danger" role="menuitem" onClick={onLogout}>Logout</button>
                </div>
              )}
            </div>
          )}
        </div>
      </nav>
    </header>
  )
}

function Sidebar() {
  return (
    <aside className="sidebar">
      <div className="section-title">Menu</div>
      <nav className="sidebar-nav">
        <NavLink to="/upload" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>Unggah Citra Daun</NavLink>
        <NavLink to="/history" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>Riwayat Prediksi</NavLink>
        <NavLink to="/reports" className={({ isActive }) => `sidebar-link${isActive ? ' active' : ''}`}>Unduh Prediksi</NavLink>
        {/* Tautan admin ditampilkan kondisional di App berdasarkan role */}
      </nav>
    </aside>
  )
}

// Mobile bottom navigation dihapus sesuai preferensi: menu dipindahkan ke atas (Navbar)

export default function App() {
  const navigate = useNavigate()
  const location = useLocation()
  // Mulai dari kondisi belum login; kemudian konfirmasi session Supabase di useEffect
  const [authed, setAuthed] = useState(false)
  // Ready flag untuk mencegah redirect ke /login sebelum sesi dicek
  const [authReady, setAuthReady] = useState(false)
  const [userName, setUserName] = useState('')
  const [userEmail, setUserEmail] = useState('')
  const [userRole, setUserRole] = useState('user')
  const [userId, setUserId] = useState('')

  // Helper: cek role admin via RPC SECURITY DEFINER agar tidak terganjal RLS
  const loadRole = async (id) => {
    if (!supabase || !id) return
    try {
      // Ganti RPC dengan pembacaan profil sendiri (diizinkan oleh RLS "read-own")
      const { data: prof, error } = await supabase
        .from('profiles')
        .select('role')
        .eq('id', id)
        .single()
      if (!error) {
        const role = String(prof?.role || '').trim().toLowerCase()
        setUserRole(role === 'admin' ? 'admin' : 'user')
      }
    } catch (_) {}
  }

  const syncProfileName = async (id, name) => {
    if (!supabase || !id || !name) return
    try {
      const { data: prof } = await supabase
        .from('profiles')
        .select('full_name')
        .eq('id', id)
        .single()
      const current = String(prof?.full_name || '').trim()
      const target = String(name || '').trim()
      if (target && current !== target) {
        await supabase.from('profiles').update({ full_name: target }).eq('id', id)
      }
    } catch (_) {}
  }

  // Jangan gunakan localStorage sebagai sumber kebenaran auth;
  // hanya percaya pada Supabase session & event auth untuk mencegah flicker.
  // Efek sinkronisasi antar-tab ditangani oleh onAuthStateChange + BroadcastChannel.

  // Terapkan preferensi font dari localStorage ke html root
  useEffect(() => {
    try {
      const pref = localStorage.getItem('font_pref') // contoh: 'font-poppins' atau 'font-jakarta'
      const root = document.documentElement
      const allowed = ['font-inter', 'font-poppins', 'font-jakarta']
      root.classList.remove(...allowed)
      if (allowed.includes(pref)) root.classList.add(pref)
    } catch (_) {}
  }, [])

  // Hydrate data user (nama & email) dari Supabase session jika tersedia
  useEffect(() => {
    (async () => {
      try {
        if (!supabase) return
        const { data } = await supabase.auth.getSession?.()
        const session = data?.session
        if (session?.user) {
          const name = session.user.user_metadata?.full_name || session.user.email || ''
          const email = session.user.email || ''
          if (name) setUserName(name)
          if (email) setUserEmail(email)
        }
      } catch (_) {}
    })()
  }, [])

  useEffect(() => {
    // Jika Supabase dikonfigurasi, sinkronkan session Supabase ke localStorage agar proteksi rute tetap bekerja
    const syncSession = async () => {
      if (!supabase) return
      const { data } = await supabase.auth.getSession()
      const access = data.session?.access_token
      if (access) {
        localStorage.setItem('auth_token', access)
        setAuthed(true)
        const u = await supabase.auth.getUser()
        const name = u?.data?.user?.user_metadata?.full_name || u?.data?.user?.email || ''
        const id = u?.data?.user?.id || ''
        setUserName(name)
        if (id) setUserId(id)
        if (id) await syncProfileName(id, name)
        // Muat role via RPC (hindari error RLS saat belum fully authed)
        if (id) await loadRole(id)
      }
      // Tandai cek sesi selesai (ada/tidaknya session)
      setAuthReady(true)
    }
    syncSession()

    if (supabase) {
      const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
        const token = session?.access_token
        if (token) {
          localStorage.setItem('auth_token', token)
          setAuthed(true)
          supabase.auth.getUser().then((u) => {
            const name = u?.data?.user?.user_metadata?.full_name || u?.data?.user?.email || ''
            const id = u?.data?.user?.id || ''
            setUserName(name)
            if (id) setUserId(id)
            if (id) {
              syncProfileName(id, name)
              loadRole(id)
            }
          }).catch(() => {})
          if (event === 'PASSWORD_RECOVERY') {
            // Arahkan ke halaman reset ketika datang dari tautan recovery
            navigate('/reset', { replace: true })
          }
        } else {
          localStorage.removeItem('auth_token')
          setAuthed(false)
          setUserName('')
          setUserRole('user')
          setUserId('')
        }
        // Setelah event auth pertama, anggap cek selesai
        setAuthReady(true)
      })
      return () => {
        sub.subscription?.unsubscribe?.()
      }
    }
  }, [])

  // Dengarkan siaran dari tab callback untuk segera sinkron
  useEffect(() => {
    try {
      const ch = new BroadcastChannel('auth')
      ch.onmessage = async (ev) => {
        if (ev?.data?.type === 'confirmed') {
          const { data } = await supabase.auth.getSession()
          const token = data.session?.access_token
          if (token) {
            localStorage.setItem('auth_token', token)
            setAuthed(true)
            try {
              const u = await supabase.auth.getUser()
              const id = u?.data?.user?.id
              if (id) await loadRole(id)
            } catch (_) {}
            navigate('/dashboard', { replace: true })
          }
        } else if (ev?.data?.type === 'recovery') {
          // Terima sinyal dari tab yang dibuka lewat email untuk berpindah ke halaman reset
          navigate('/reset', { replace: true })
        }
      }
      return () => ch.close()
    } catch (_) {
      // BroadcastChannel mungkin tidak tersedia di beberapa environment; abaikan.
    }
  }, [navigate])

  // Saat jendela kembali fokus, refresh role (berguna setelah role diubah di server)
  useEffect(() => {
    const onFocus = async () => {
      if (!supabase) return
      try {
        const u = await supabase.auth.getUser()
        const id = u?.data?.user?.id
        if (id) await loadRole(id)
      } catch (_) {}
    }
    window.addEventListener('focus', onFocus)
    const onVisible = async () => {
      if (document.visibilityState === 'visible') {
        try {
          const u = await supabase.auth.getUser()
          const id = u?.data?.user?.id
          if (id) await loadRole(id)
        } catch (_) {}
      }
    }
    document.addEventListener('visibilitychange', onVisible)
    return () => window.removeEventListener('focus', onFocus)
  }, [])

  const handleLoginSuccess = (data) => {
    const token = data?.session?.access_token || data?.token || 'local_auth_token'
    localStorage.setItem('auth_token', token)
    setAuthed(true)
    const name = data?.session?.user?.user_metadata?.full_name || data?.session?.user?.email || ''
    if (name) setUserName(name)
    const idNow = data?.session?.user?.id || ''
    if (idNow && name) syncProfileName(idNow, name)
    const email = data?.session?.user?.email || data?.user?.email || ''
    if (email) setUserEmail(email)
    navigate('/dashboard')
  }

  const handleLogout = async () => {
    try { await supabase?.auth?.signOut?.() } catch (_) {}
    localStorage.removeItem('auth_token')
    setAuthed(false)
    setUserEmail('')
    navigate('/login', { replace: true })
  }

  const Protected = ({ children }) => authed ? children : (authReady ? <Navigate to="/login" replace /> : <div className="container"><div className="muted">Memeriksa sesi…</div></div>)
  return (
    <div className="app">
      {(() => {
        const isAuthPage = (
          location.pathname.startsWith('/login') ||
          location.pathname.startsWith('/signup') ||
          location.pathname.startsWith('/reset') ||
          location.pathname.startsWith('/forgot') ||
          location.pathname.startsWith('/auth/callback')
        )
        return !isAuthPage ? (
          <Navbar authed={authed} onLogout={handleLogout} userName={userName} userEmail={userEmail} userRole={userRole} />
        ) : null
      })()}
      {authed ? (
        <div className="layout">
          {/* Sidebar di-nonaktifkan agar menu konsisten di bagian atas */}
          {/* <div className="hidden md:block"><Sidebar /></div> */}
          <main className="content">
            <Routes>
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
              <Route path="/dashboard" element={<Protected><Dashboard /></Protected>} />
              {/* Izinkan akses halaman Login meski sudah login (sesuai kebutuhan pengguna) */}
              <Route path="/login" element={<Login onSuccess={handleLoginSuccess} />} />
              <Route path="/auth/callback" element={<AuthCallback />} />
              <Route path="/upload" element={<Protected><Upload /></Protected>} />
              <Route path="/live" element={<Protected><LiveDetect /></Protected>} />
              <Route path="/results" element={<Protected><Results /></Protected>} />
              <Route path="/history" element={<Protected><History /></Protected>} />
              <Route path="/reports" element={<Protected><Reports /></Protected>} />
              {String(userRole).trim().toLowerCase() === 'admin' && (
                <Route path="/admin/users" element={<Protected><UserManagement /></Protected>} />
              )}
              <Route path="/reset" element={<ResetPassword />} />
              <Route path="/forgot" element={<ForgotPassword />} />
              <Route path="/signup" element={<Navigate to="/dashboard" replace />} />
              {/* Fallback untuk rute apapun saat sudah login */}
              <Route path="*" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </main>
        </div>
      ) : (
        authReady ? (
          <main className="container">
            <Routes>
              <Route path="/" element={<Navigate to="/login" replace />} />
              <Route path="/dashboard" element={<Navigate to="/login" replace />} />
              <Route path="/auth/callback" element={<AuthCallback />} />
              <Route path="/login" element={<Login onSuccess={handleLoginSuccess} />} />
              <Route path="/signup" element={<SignUp onSuccess={handleLoginSuccess} />} />
              <Route path="/reset" element={<ResetPassword />} />
              <Route path="/forgot" element={<ForgotPassword />} />
              {/* Redirect eksplisit untuk rute terproteksi saat belum authed */}
              <Route path="/upload" element={<Navigate to="/login" replace />} />
              <Route path="/results" element={<Navigate to="/login" replace />} />
              <Route path="/history" element={<Navigate to="/login" replace />} />
              <Route path="/reports" element={<Navigate to="/login" replace />} />
              <Route path="/admin/users" element={<Navigate to="/login" replace />} />
              {/* Fallback: semua rute lain menuju /login */}
              <Route path="*" element={<Navigate to="/login" replace />} />
            </Routes>
          </main>
        ) : (
          <main className="container">
            <div className="muted">Memeriksa sesi…</div>
          </main>
        )
      )}
    </div>
  )
}