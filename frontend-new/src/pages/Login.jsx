import { useState } from 'react'
import { Link } from 'react-router-dom'
import { supabase } from '../lib/supabaseClient'
import logoModern from '../assets/logo-modern.svg'

// Ikon SVG untuk toggle visibilitas password
function EyeIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M1 12c2.5-5 7.5-8 11-8s8.5 3 11 8c-2.5 5-7.5 8-11 8s-8.5-3-11-8z" fill="none" stroke="currentColor" strokeWidth="2"/>
      <circle cx="12" cy="12" r="3" fill="currentColor"/>
    </svg>
  )
}

function EyeOffIcon({ size = 20 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M1 12c2.5-5 7.5-8 11-8s8.5 3 11 8c-2.5 5-7.5 8-11 8s-8.5-3-11-8z" fill="none" stroke="currentColor" strokeWidth="2"/>
      <circle cx="12" cy="12" r="3" fill="currentColor"/>
      <line x1="4" y1="4" x2="20" y2="20" stroke="currentColor" strokeWidth="2"/>
    </svg>
  )
}

function MailArrowIcon({ size = 18 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true">
      <path d="M3 6h18v12H3z" fill="none" stroke="currentColor" strokeWidth="2"/>
      <path d="M3 7l9 6 9-6" fill="none" stroke="currentColor" strokeWidth="2"/>
      <path d="M14 16h5m0 0l-2 2m2-2l-2-2" fill="none" stroke="currentColor" strokeWidth="2"/>
    </svg>
  )
}

export default function Login({ onSuccess }) {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [resending, setResending] = useState(false)

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)
    try {
      if (!supabase) {
        if (!email || !password) throw new Error('Isi email dan password')
        const token = 'local_demo_' + Date.now()
        onSuccess?.({ token })
        return
      }
      const { data, error } = await supabase.auth.signInWithPassword({ email, password })
      if (error) throw error
      onSuccess?.(data)
    } catch (err) {
      setError(err.message || 'Login gagal')
    } finally {
      setLoading(false)
    }
  }

  const handleResendVerification = async () => {
    setError('')
    setInfo('')
    if (!email) {
      setError('Masukkan email untuk kirim ulang verifikasi')
      return
    }
    if (!supabase) {
      setInfo('Mode lokal: pengiriman email verifikasi tidak tersedia.')
      return
    }
    setResending(true)
    try {
      const { error } = await supabase.auth.resend({
        type: 'signup',
        email,
        options: { emailRedirectTo: `${window.location.origin}/auth/callback` },
      })
      if (error) throw error
      setInfo('Email verifikasi dikirim ulang. Periksa inbox atau folder spam.')
    } catch (err) {
      setError(err.message || 'Gagal mengirim ulang verifikasi')
    } finally {
      setResending(false)
    }
  }

  return (
    <div className="auth-wrap">
      <section className="card auth-card green">
        <div className="auth-brand">
          <img src={logoModern} alt="logo" className="brand-logo" />
          <div className="brand-name">ChiliVar</div>
        </div>
        <h1 className="card-title center">Login</h1>
        <p className="muted center">Masuk untuk mengakses fitur klasifikasi.</p>
        <form onSubmit={handleSubmit} className="form">
          <label>
            Email
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          <label>
            Password
            <div className="password-field">
              <input
                type={showPw ? 'text' : 'password'}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
              <button
                type="button"
                className="toggle-eye"
                aria-label={showPw ? 'Sembunyikan password' : 'Tampilkan password'}
                aria-pressed={showPw}
                onClick={() => setShowPw((v) => !v)}
                title={showPw ? 'Sembunyikan' : 'Tampilkan'}
              >
                {showPw ? <EyeOffIcon /> : <EyeIcon />}
              </button>
            </div>
          </label>
        <div className="action-row" style={{ justifyContent: 'flex-end' }}>
          <a href="/forgot" className="link-green">Lupa kata sandi?</a>
        </div>
        {error && <div className="error">{error}</div>}
        {info && <div className="info" role="status">{info}</div>}
        <button type="submit" className="btn block" disabled={loading}>
          {loading ? 'Memproses…' : 'Masuk'}
        </button>
        {/* Duplikat aksi dihapus untuk merapikan UI */}
      </form>
      <div className="center muted" style={{ marginTop: 10 }}>
        Belum punya akun?{' '}
        <Link to="/signup" className="link-green">Sign Up</Link>
      </div>
      {!supabase && (
        <div className="warning center">Supabase belum dikonfigurasi (.env). Login lokal aktif: isi email dan password apa saja untuk masuk.</div>
      )}
      </section>
    </div>
  )
}