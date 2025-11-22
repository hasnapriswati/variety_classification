import { useState } from 'react'
import { supabase } from '../lib/supabaseClient'

export default function ForgotPassword() {
  const [email, setEmail] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [info, setInfo] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setInfo('')
    if (!email) {
      setError('Isi email untuk mengirim tautan reset')
      return
    }
    setLoading(true)
    try {
      if (!supabase) {
        setInfo('Mode lokal: fitur reset password membutuhkan Supabase terkonfigurasi.')
        return
      }
      const { error } = await supabase.auth.resetPasswordForEmail(email, {
        redirectTo: `${window.location.origin}/reset`,
      })
      if (error) throw error
      setInfo('Tautan reset dikirim. Silakan cek email Anda dan ikuti tautan untuk mengganti password.')
    } catch (err) {
      setError(err.message || 'Gagal mengirim tautan reset')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="auth-wrap">
      <section className="card auth-card">
        <h1 className="card-title center">Lupa Kata Sandi</h1>
        <p className="muted center">Masukkan email Anda untuk menerima tautan reset.</p>
        <form onSubmit={handleSubmit} className="form">
          <label>
            Email
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required />
          </label>
          {error && <div className="error" role="alert">{error}</div>}
          {info && <div className="info" role="status">{info}</div>}
          <button type="submit" className="btn block" disabled={loading}>
            {loading ? 'Mengirim…' : 'Kirim Tautan Reset'}
          </button>
        </form>
        <div className="center muted" style={{ marginTop: 10 }}>
          <a href="/login">Kembali ke Login</a>
        </div>
        {!supabase && (
          <div className="warning center">Supabase belum dikonfigurasi (.env). Fitur reset password memerlukan integrasi Supabase.</div>
        )}
      </section>
    </div>
  )
}