import { useEffect, useMemo, useState } from 'react'
import { supabase } from '../lib/supabaseClient'

export default function UserManagement() {
  // Ikon kecil untuk tombol aksi
  const CrownIcon = ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true"><path d="M3 18h18l-2-9-5 4-4-6-5 7-2 4z" fill="currentColor"/></svg>
  )
  const BanIcon = ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="9" fill="none" stroke="currentColor" strokeWidth="2"/><path d="M7 7l10 10" stroke="currentColor" strokeWidth="2"/></svg>
  )
  const RestoreIcon = ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true"><path d="M12 5v4l4-4M6 12a6 6 0 1012 0 6 6 0 10-12 0z" fill="none" stroke="currentColor" strokeWidth="2"/></svg>
  )
  const Spinner = ({ size = 16 }) => (
    <svg width={size} height={size} viewBox="0 0 24 24" aria-hidden="true" className="spin"><circle cx="12" cy="12" r="10" fill="none" stroke="currentColor" strokeWidth="2" opacity="0.25"/><path d="M12 2a10 10 0 000 20" fill="none" stroke="currentColor" strokeWidth="2"/></svg>
  )
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [users, setUsers] = useState([])
  const [busyId, setBusyId] = useState('')
  const [query, setQuery] = useState('')
  const [searchText, setSearchText] = useState('')
  const [lastMap, setLastMap] = useState({})
  // Debounce: sinkronkan searchText ke query agar UX lebih halus
  useEffect(() => {
    const t = setTimeout(() => setQuery(searchText), 250)
    return () => clearTimeout(t)
  }, [searchText])

  const load = async () => {
    setLoading(true)
    setError('')
    try {
      const { data, error } = await supabase
        .from('profiles')
        .select('id, full_name, role, status, created_at')
        .order('created_at', { ascending: false })
      if (error) throw error
      setUsers(data || [])
      try {
        const ids = (data || []).map(u => u.id).filter(Boolean)
        if (ids.length) {
          const { data: preds } = await supabase
            .from('predictions')
            .select('user_id, created_at')
            .in('user_id', ids)
            .order('created_at', { ascending: false })
          const map = {}
          for (const p of (preds || [])) {
            const uid = p.user_id
            if (uid && !map[uid]) map[uid] = p.created_at
          }
          setLastMap(map)
        } else {
          setLastMap({})
        }
      } catch (_) {}
    } catch (e) {
      setError(e?.message || 'Gagal memuat profil pengguna')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  const updateRole = async (id, role) => {
    if (!window.confirm(`Ubah peran pengguna menjadi \"${role}\"?`)) return
    setBusyId(id)
    setError('')
    try {
      const { error } = await supabase.from('profiles').update({ role }).eq('id', id)
      if (error) throw error
      await load()
    } catch (e) {
      setError(e?.message || 'Gagal memperbarui peran')
    } finally {
      setBusyId('')
    }
  }

  const updateStatus = async (id, status) => {
    const label = status === 'suspended' ? 'Nonaktifkan akun ini?' : 'Pulihkan akun ini?'
    if (!window.confirm(label)) return
    setBusyId(id)
    setError('')
    try {
      const { error } = await supabase.from('profiles').update({ status }).eq('id', id)
      if (error) throw error
      await load()
    } catch (e) {
      setError(e?.message || 'Gagal memperbarui status')
    } finally {
      setBusyId('')
    }
  }

  // Daftar terfilter sederhana berdasar nama atau ID
  const filteredUsers = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return users
    return users.filter(u => (
      String(u.full_name || '').toLowerCase().includes(q) ||
      String(u.id || '').toLowerCase().includes(q)
    ))
  }, [users, query])

  return (
    <div className="container">
      <h1 className="page-title">Manajemen Pengguna</h1>

      <div className="toolbar" style={{ display: 'flex', gap: 8, alignItems: 'center', marginBottom: 12 }}>
        <div className="search-field" style={{ flex: 1, minWidth: 260 }}>
          <span className="search-icon" aria-hidden>🔎</span>
          <input
            type="text"
            placeholder="Cari nama atau ID"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            className="search-input"
            aria-label="Filter pengguna"
          />
          {!!searchText && (
            <button className="search-clear" aria-label="Bersihkan pencarian" onClick={() => setSearchText('')}>✕</button>
          )}
        </div>
        <button className="filter-button" onClick={load} disabled={loading}>Refresh</button>
      </div>

      {error && (
        <div className="alert error" role="alert">{error}</div>
      )}
      {loading ? (
        <div className="loading">Memuat pengguna…</div>
      ) : (
        <div className="card">
          <div className="table-responsive">
            <table className="table">
              <thead>
                <tr>
                  <th style={{minWidth: 160}}>Nama</th>
                  <th style={{minWidth: 180}}>ID</th>
                  <th>Peran</th>
                  <th>Status</th>
                  <th>Terakhir aktivitas</th>
                  <th>Dibuat</th>
                  <th className="actions">Aksi</th>
                </tr>
              </thead>
              <tbody>
                {filteredUsers.map(u => (
                  <tr key={u.id}>
                    <td>{u.full_name || '-'}</td>
                    <td><code>{u.id}</code></td>
                    <td>
                      {String(u.role).trim().toLowerCase() === 'admin' ? (
                        <span className="status-chip admin" title="Admin"><span className="dot" /><span>Admin</span></span>
                      ) : (
                        <span className="status-chip user" title="Pengguna"><span className="dot" /><span>User</span></span>
                      )}
                    </td>
                    <td>
                      {String(u.status).trim().toLowerCase() === 'suspended' ? (
                        <span className="status-chip suspended" title="Disuspend"><span className="dot" /><span>Suspended</span></span>
                      ) : (
                        <span className="status-chip active" title="Akun aktif"><span className="dot" /><span>Aktif (akun)</span></span>
                      )}
                    </td>
                    
                    <td>{lastMap[u.id] ? new Date(lastMap[u.id]).toLocaleString() : '-'}</td>
                    <td>{new Date(u.created_at).toLocaleString()}</td>
                    <td className="actions">
                      <div className="btn-group" aria-label="Aksi pengguna">
                        {u.role !== 'admin' ? (
                          <button
                            className="btn outline sm icon"
                            disabled={busyId===u.id}
                            onClick={() => updateRole(u.id, 'admin')}
                            aria-label="Jadikan admin"
                            title="Jadikan admin"
                          >
                            {busyId===u.id ? <Spinner /> : <CrownIcon />}
                            <span>Jadikan Admin</span>
                          </button>
                        ) : (
                          <span className="badge success" title="Sudah admin">Admin</span>
                        )}
                        {u.status !== 'suspended' ? (
                          <button
                            className="btn danger outline sm icon"
                            disabled={busyId===u.id}
                            onClick={() => updateStatus(u.id, 'suspended')}
                            aria-label="Nonaktifkan akun"
                            title="Nonaktifkan akun"
                          >
                            {busyId===u.id ? <Spinner /> : <BanIcon />}
                            <span>Nonaktifkan</span>
                          </button>
                        ) : (
                          <button
                            className="btn success outline sm icon"
                            disabled={busyId===u.id}
                            onClick={() => updateStatus(u.id, 'active')}
                            aria-label="Pulihkan akun"
                            title="Pulihkan akun"
                          >
                            {busyId===u.id ? <Spinner /> : <RestoreIcon />}
                            <span>Pulihkan</span>
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
                {filteredUsers.length === 0 && (
                  <tr><td colSpan={5} style={{textAlign:'center', padding: '24px'}}>Tidak ada pengguna.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}