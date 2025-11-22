export function formatDate(input) {
  const d = new Date(input)
  if (Number.isNaN(d.getTime())) return '-'
  const day = String(d.getDate()).padStart(2, '0')
  const month = String(d.getMonth() + 1).padStart(2, '0')
  const year = d.getFullYear()
  return `${day}/${month}/${year}`
}

export function isValidDate(input) {
  const d = new Date(input)
  return !Number.isNaN(d.getTime())
}