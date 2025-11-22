import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabaseClient'

export default function AuthCallback() {
  const navigate = useNavigate()
  const [status, setStatus] = useState('Menyambungkan akun…')

  useEffect(() => {
    const run = async () => {
      try {
        if (!supabase) {
          navigate('/login', { replace: true })
          return
        }
        const params = new URLSearchParams(window.location.search)
        const code = params.get('code')
        const errorDesc = params.get('error_description')
        const tokenHash = params.get('token_hash')
        const typeParam = params.get('type') // bisa 'email' atau 'signup' tergantung template

        if (errorDesc) {
          setStatus('Gagal konfirmasi: ' + errorDesc)
          navigate('/login', { replace: true })
          return
        }

        // 1) PKCE / OAuth: ada ?code=
        if (code) {
          await supabase.auth.exchangeCodeForSession(code)
        }

        // 2) Email confirmation / Magic link dengan token_hash
        // Supabase merekomendasikan verifyOtp dengan type 'email' untuk token_hash.
        if (!code && tokenHash) {
          const type = typeParam === 'signup' ? 'email' : (typeParam || 'email')
          const { error: vErr } = await supabase.auth.verifyOtp({ token_hash: tokenHash, type })
          if (vErr) {
            setStatus('Gagal verifikasi token: ' + (vErr?.message || 'Unknown error'))
          }
        }

        const { data } = await supabase.auth.getSession()
        if (data?.session?.access_token) {
          localStorage.setItem('auth_token', data.session.access_token)
          // Beritahu tab utama bahwa auth selesai
          try {
            const ch = new BroadcastChannel('auth')
            ch.postMessage({ type: 'confirmed' })
            ch.close()
          } catch (_) {}

          // Fallback: jika tab utama tidak terbuka, arahkan tab ini ke aplikasi.
          const hasOpener = !!window.opener
          const sameOrigin = window.location.origin === (document.referrer ? new URL(document.referrer).origin : window.location.origin)
          if (!hasOpener || !sameOrigin) {
            navigate('/dashboard', { replace: true })
            return
          }

          // Jika ada tab utama, coba tutup tab callback.
          setStatus('Konfirmasi berhasil. Anda akan otomatis masuk di tab aplikasi. Tab ini bisa ditutup.')
          setTimeout(() => {
            try {
              window.open('', '_self')
              window.close()
            } catch (_) {}
          }, 300)
        } else {
          // Jika email confirmation tidak menghasilkan sesi, arahkan ke login
          setStatus('Konfirmasi selesai. Jika belum otomatis masuk, silakan login di aplikasi.')
          // Sebagai alternatif, arahkan ke /login agar pengguna dapat masuk manual.
          setTimeout(() => {
            navigate('/login', { replace: true })
          }, 1000)
        }
      } catch (err) {
        setStatus('Terjadi kesalahan: ' + (err?.message || 'Unknown error'))
      }
    }
    run()
  }, [navigate])

  return (
    <div className="auth-wrap">
      <section className="card auth-card">
        <h1 className="card-title center">Menghubungkan…</h1>
        <p className="muted center">{status}</p>
      </section>
    </div>
  )
}