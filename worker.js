addEventListener('fetch', event => {
  event.respondWith(handleRequest(event.request))
})

async function handleRequest(request) {
  const API_URL = 'https://api.casinoscores.com/svc-evolution-game-events/api/xxxtremelightningroulette/latest'
  const resp = await fetch(API_URL, {
    headers: { 'User-Agent': 'Mozilla/5.0' }
  })

  const contentType = resp.headers.get('content-type') || ''
  const body = contentType.includes('application/json')
    ? JSON.stringify(await resp.json())
    : await resp.text()

  return new Response(body, {
    status: resp.status,
    headers: { 'Content-Type': contentType }
  })
}
