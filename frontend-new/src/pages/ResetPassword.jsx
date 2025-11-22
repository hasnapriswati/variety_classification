import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabaseClient'

// Ikon toggle visibilitas password (konsisten dengan Login/SignUp)
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

export default function ResetPassword() {
  const navigate = useNavigate()
  const [password, setPassword] = useState('')
  const [confirm, setConfirm] = useState('')
  const [showPw, setShowPw] = useState(false)
  const [showConfirmPw, setShowConfirmPw] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')

  // Kirim sinyal ke tab aplikasi yang sudah terbuka agar berpindah ke /reset,
  // lalu coba tutup tab baru (jika diizinkan oleh browser/email client).
  useEffect(() => {
    try {
      const ch = new BroadcastChannel('auth')
      ch.postMessage({ type: 'recovery' })
      // Upaya menutup tab jika diizinkan (umumnya diblokir untuk tab dari email)
      setTimeout(() => {
        try { window.close() } catch (_) {}
      }, 500)
      return () => ch.close()
    } catch (_) {}
  }, [])

  useEffect(() => {
    // Jika Supabase mendeteksi recovery, arahkan pengguna ke halaman ini
    if (!supabase) return
    const { data: sub } = supabase.auth.onAuthStateChange((event, session) => {
      if (event === 'PASSWORD_RECOVERY' && session?.access_token) {
        setInfo('Silakan masukkan password baru Anda.')
      }
    })
    return () => sub.subscription?.unsubscribe?.()
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setInfo('')
    if (password.length < 6) {
      setError('Password minimal 6 karakter')
      return
    }
    if (password !== confirm) {
      setError('Konfirmasi password tidak sama')
      return
    }
    setLoading(true)
    try {
      if (!supabase) {
        setInfo('Mode lokal: reset password membutuhkan Supabase terkonfigurasi.')
        return
      }
      const { data, error } = await supabase.auth.updateUser({ password })
      if (error) throw error
      setInfo('Password berhasil diperbarui. Anda akan diarahkan ke halaman dashboard.')
      navigate('/dashboard', { replace: true })
    } catch (err) {
      setError(err.message || 'Gagal memperbarui password')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-wrap">
      <section className="card auth-card">
        <h1 className="card-title center">Reset Password</h1>
        <p className="muted center">Masukkan password baru untuk akun Anda.</p>
        <form onSubmit={handleSubmit} className="form">
          <label>
            Password Baru
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
            Konfirmasi Password Baru
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
                aria-label={showConfirmPw ? 'Sembunyikan password' : 'Tampilkan password'}
                aria-pressed={showConfirmPw}
                onClick={() => setShowConfirmPw((v) => !v)}
                title={showConfirmPw ? 'Sembunyikan' : 'Tampilkan'}
              >
                {showConfirmPw ? <EyeOffIcon /> : <EyeIcon />}
              </button>
            </div>
          </label>
          {error && <div className="error" role="alert">{error}</div>}
          {info && <div className="info" role="status">{info}</div>}
          <button type="submit" className="btn block" disabled={loading}>
            {loading ? 'Memproses…' : 'Perbarui Password'}
          </button>
        </form>
        {!supabase && (
          <div className="warning center">Supabase belum dikonfigurasi (.env). Fitur reset password memerlukan integrasi Supabase.</div>
        )}
      </section>
    </div>
  )
}