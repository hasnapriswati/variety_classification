import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
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

export default function SignUp({ onSuccess }) {
  const navigate = useNavigate()
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [showConfirmPw, setShowConfirmPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')
  const [resending, setResending] = useState(false)

  // Jika konfirmasi email selesai dan sesi aktif, App akan mengubah state authed
  // Di sini kita hanya menampilkan informasi, tanpa redirect ke login.
  useEffect(() => {
    if (!supabase) return
    const { data: sub } = supabase.auth.onAuthStateChange((_evt, session) => {
      if (session?.access_token) {
        onSuccess?.({ session })
        navigate('/dashboard', { replace: true })
      }
    })
    return () => sub.subscription?.unsubscribe?.()
  }, [navigate, onSuccess])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    if (!fullName.trim()) {
      setError('Nama lengkap wajib diisi')
      return
    }
    if (password !== confirm) {
      setError('Konfirmasi password tidak sama')
      return
    }
    setLoading(true)
    try {
      if (!supabase) {
        // Mode lokal bila Supabase belum dikonfigurasi
        if (!email || !password) throw new Error('Isi email dan password')
        const token = 'local_signup_' + Date.now()
        onSuccess?.({ token })
        navigate('/dashboard')
        return
      }
      const { data, error } = await supabase.auth.signUp({
        email,
        password,
        options: {
          data: { full_name: fullName },
          emailRedirectTo: `${window.location.origin}/auth/callback`,
        },
      })
      if (error) throw error
      // Jika verifikasi email diaktifkan, Supabase tidak mengembalikan session.
      // Tampilkan info agar pengguna konfirmasi email, tetap di halaman ini.
      if (!data?.session) {
        setInfo('Registrasi berhasil. Silakan cek email Anda untuk konfirmasi. Setelah dikonfirmasi, Anda akan otomatis masuk tanpa perlu login ulang.')
        return
      }
      // Bila session langsung tersedia (email confirmation dimatikan), masuk otomatis
      onSuccess?.(data)
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setError(err.message || 'Sign up gagal')
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
        <h1 className="card-title center">Buat Akun</h1>
        <p className="muted center">Daftar untuk mengakses fitur klasifikasi.</p>
        <form onSubmit={handleSubmit} className="form">
          <label>
            Nama Lengkap
            <input type="text" value={fullName} onChange={(e) => setFullName(e.target.value)} required />
          </label>
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
          <label>
            Konfirmasi Password
            <div className="password-field">
              <input
                type={showConfirmPw ? 'text' : 'password'}
                value={confirm}
                onChange={(e) => setConfirm(e.target.value)}
                required
              />
              <button
                type="button"
                className="toggle-eye"
                aria-label={showConfirmPw ? 'Sembunyikan konfirmasi password' : 'Tampilkan konfirmasi password'}
                aria-pressed={showConfirmPw}
                onClick={() => setShowConfirmPw((v) => !v)}
                title={showConfirmPw ? 'Sembunyikan' : 'Tampilkan'}
              >
                {showConfirmPw ? <EyeOffIcon /> : <EyeIcon />}
              </button>
            </div>
          </label>
          {error && <div className="error">{error}</div>}
          {info && <div className="info" role="status">{info}</div>}
          <div className="action-row" style={{ marginTop: info ? 8 : 0 }}>
            <button
              type="button"
              className="btn ghost sm icon"
              onClick={handleResendVerification}
              disabled={resending}
              title="Kirim ulang email konfirmasi untuk alamat di kolom Email"
            >
              <MailArrowIcon /> {resending ? 'Mengirim…' : 'Kirim ulang verifikasi'}
            </button>
          </div>
          <button type="submit" className="btn block" disabled={loading}>
            {loading ? 'Memproses…' : 'Daftar'}
          </button>
        </form>
        <div className="center muted" style={{ marginTop: 10 }}>
          Sudah punya akun?{' '}
          <Link to="/login" className="link-green">Login</Link>
        </div>
        {!supabase && (
          <div className="warning center">Supabase belum dikonfigurasi (.env). Mode lokal aktif: isi email dan password apa saja untuk daftar.</div>
        )}
      </section>
    </div>
  )
}