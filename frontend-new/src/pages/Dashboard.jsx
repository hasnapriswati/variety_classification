import { useEffect, useMemo, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { supabase } from '../lib/supabaseClient'

function formatPercent(num) {
  if (typeof num !== 'number' || isNaN(num)) return '-'
  return `${(num * 100).toFixed(2)}%`
}

function confidenceValue(item) {
  if (!item) return 0
  if (typeof item.confidence === 'number' && !isNaN(item.confidence)) return item.confidence
  const s = item.confidence_percentage
  if (typeof s === 'string') {
    const m = s.match(/([0-9]+(?:\.[0-9]+)?)/)
    if (m) {
      const v = parseFloat(m[1]) / 100
      return isNaN(v) ? 0 : Math.min(v, 0.9999)
    }
  }
  return 0
}

function confidenceLevel(item) {
  const v = confidenceValue(item)
  if (v >= 0.95) return 'Sangat yakin'
  if (v >= 0.90) return 'Cukup yakin'
  return 'Tidak pasti'
}

export default function Dashboard() {
  const navigate = useNavigate()
  const [items, setItems] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [userId, setUserId] = useState('')
  const [isAdmin, setIsAdmin] = useState(false)
  const [profilesMap, setProfilesMap] = useState({})

  useEffect(() => {
    const run = async () => {
      try {
        if (!supabase) {
          setItems([])
          setLoading(false)
          return
        }
        const { data: u } = await supabase.auth.getUser()
        const id = u?.user?.id || ''
        setUserId(id)
        let admin = false
        if (id) {
          try {
            const { data: prof } = await supabase.from('profiles').select('role').eq('id', id).single()
            admin = String(prof?.role || '').trim().toLowerCase() === 'admin'
            setIsAdmin(admin)
          } catch {}
        }
        let q = supabase.from('predictions').select('*').order('created_at', { ascending: false })
        if (!admin) q = q.eq('user_id', id)
        q = q.limit(admin ? 200 : 50)
        const { data, error } = await q
        if (error) throw error
        const arr = Array.isArray(data) ? data : []
        setItems(arr)
        if (admin) {
          const ids = Array.from(new Set(arr.map(it => it.user_id).filter(Boolean)))
          if (ids.length) {
            try {
              const { data: profs } = await supabase.from('profiles').select('id, full_name, role').in('id', ids)
              const map = {}
              for (const p of (profs || [])) map[p.id] = { full_name: p.full_name || '', role: p.role || 'user' }
              setProfilesMap(map)
            } catch {}
          }
        }
      } catch (e) {
        setError(e.message || 'Gagal memuat dashboard')
      } finally {
        setLoading(false)
      }
    }
    run()
  }, [])

  const stats = useMemo(() => {
    const total = items.length
    let sangat = 0, cukup = 0, tidak = 0
    for (const it of items) {
      const lvl = confidenceLevel(it)
      if (lvl === 'Sangat yakin') sangat++
      else if (lvl === 'Cukup yakin') cukup++
      else tidak++
    }
    const latest = items[0] || null
    const avg = total ? (items.reduce((s, it) => s + confidenceValue(it), 0) / total) : 0
    const hi = items.filter(it => confidenceValue(it) >= 0.95).length
    const mid = items.filter(it => confidenceValue(it) >= 0.90 && confidenceValue(it) < 0.95).length
    const low = items.filter(it => confidenceValue(it) < 0.90).length
    const issuesMap = new Map()
    for (const it of items) {
      const list = (it?.measurement_quality?.issues) || (it?.morphology_info?.issues) || []
      if (Array.isArray(list)) {
        for (const k of list) {
          if (!k) continue
          issuesMap.set(k, (issuesMap.get(k) || 0) + 1)
        }
      }
    }
    const issues = Array.from(issuesMap.entries()).sort((a,b) => b[1]-a[1]).slice(0,5)
    const days = Array(7).fill(0)
    const now = Date.now()
    for (const it of items) {
      const t = new Date(it.created_at).getTime()
      const diff = Math.floor((now - t) / (24*3600*1000))
      if (diff >= 0 && diff < 7) days[6 - diff]++
    }
    const usersCount = isAdmin ? Array.from(new Set(items.map(it => it.user_id).filter(Boolean))).length : 0
    return { total, sangat, cukup, tidak, latest, avg, hi, mid, low, issues, days, usersCount }
  }, [items, isAdmin])

  const onGoUpload = () => navigate('/upload')
  const onGoLive = () => navigate('/live')

  return (
    <section className="card">
      <h1 className="card-title">{isAdmin ? 'Dashboard Admin' : 'Dashboard'}</h1>
      {loading ? (
        <div className="muted">Memuat ringkasan…</div>
      ) : error ? (
        <div className="error">{error}</div>
      ) : (
        <>
          <div className={`report-stats ${isAdmin ? 'compact4' : 'compact3'}`}>
            <div className="stat-card green">
              <div className="stat-value">{stats.total}</div>
              <div className="stat-label">{isAdmin ? 'Unggahan (Semua)' : 'Unggahan'}</div>
            </div>
            <div className="stat-card yellow">
              <div className="stat-value">{stats.sangat + stats.cukup}</div>
              <div className="stat-label">Prediksi yakin</div>
            </div>
            <div className="stat-card red">
              <div className="stat-value">{stats.tidak}</div>
              <div className="stat-label">Tidak pasti</div>
            </div>
            {isAdmin && (
              <div className="stat-card gray">
                <div className="stat-value">{stats.usersCount}</div>
                <div className="stat-label">Pengguna aktif</div>
              </div>
            )}
          </div>

          <div className="section-grid two" style={{ marginTop: 12 }}>
            <div className="section">
              <div className="section-title">Rata-rata Confidence</div>
              <div className="progress">
                <div className="progress-bar" style={{ width: `${(stats.avg || 0) * 100}%` }} />
              </div>
              <div className="metrics" style={{ marginTop: 10 }}>
                <div className="metric"><span className="muted">≥95%</span><strong>{stats.hi}</strong></div>
                <div className="metric"><span className="muted">90–95%</span><strong>{stats.mid}</strong></div>
                <div className="metric"><span className="muted">&lt;90%</span><strong>{stats.low}</strong></div>
              </div>
            </div>
            <div className="section">
              <div className="section-title">Tren 7 Hari</div>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(7, 1fr)', gap: 8, alignItems: 'end', height: 80 }}>
                {stats.days.map((v, i) => {
                  const max = Math.max(1, ...stats.days)
                  const h = Math.round((v / max) * 70)
                  const labels = ['Jum','Sab','Min','Sen','Sel','Rab','Kam']
                  return (
                    <div key={i} style={{ display: 'grid', justifyItems: 'center', gap: 6 }}>
                      <div style={{ width: '14px', height: `${h}px`, background: '#2F7E79', borderRadius: '6px' }} />
                      <div className="small muted" style={{ textAlign: 'center' }}>{labels[i]}</div>
                    </div>
                  )
                })}
              </div>
            </div>
          </div>

          

          <div className="section" style={{ marginTop: 12 }}>
            <div className="section-title">Aktivitas Terakhir</div>
            {items.length === 0 ? (
              <div className="muted">Tidak ada data.</div>
            ) : (
              <div className="history-list" role="list">
                {items.slice(0, 5).map((it) => (
                  <div key={it.id} className="history-item" role="listitem">
                    <div className="history-icon">🌿</div>
                    <div>
                      <div className="history-top">
                        <div className="history-pred">{it.predicted_class || it.variety || '-'}</div>
                        <div className="muted">{it.confidence_percentage || formatPercent(confidenceValue(it))}</div>
                      </div>
                      <div className="history-meta">
                        <span>{new Date(it.created_at).toLocaleString()}</span>
                        {isAdmin && it.user_id && (
                          <span>· {profilesMap[it.user_id]?.full_name || it.user_id}</span>
                        )}
                        {Array.isArray(it?.morphology_info?.issues) && it.morphology_info.issues.length > 0 && (
                          <span>· {it.morphology_info.issues[0]}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </section>
  )
}