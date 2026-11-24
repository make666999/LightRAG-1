import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import type { LightragStatus } from '@/api/lightrag'

type StateModule = typeof import('./state')

const storageMock = () => {
  const data = new Map<string, string>()

  return {
    getItem: (key: string) => data.get(key) ?? null,
    setItem: (key: string, value: string) => {
      data.set(key, value)
    },
    removeItem: (key: string) => {
      data.delete(key)
    },
    clear: () => {
      data.clear()
    }
  }
}

const b64url = (value: string) =>
  Buffer.from(value, 'utf8')
    .toString('base64')
    .replace(/\+/g, '-')
    .replace(/\//g, '_')
    .replace(/=+$/, '')

const jwt = (payload: unknown) =>
  `${b64url('{"alg":"HS256"}')}.${b64url(JSON.stringify(payload))}.signature`

interface Deferred<T = unknown> {
  promise: Promise<T>
  resolve: (value: T) => void
}
const deferred = <T = unknown>(): Deferred<T> => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((r) => {
    resolve = r
  })
  return { promise, resolve }
}

type HealthResponse =
  | LightragStatus
  | { status: 'healthy'; pipeline_busy: boolean; api_docs_available?: boolean }
  | { status: 'error'; message: string }

// checkHealth is swapped per step; the store must never see the network.
let healthResponse: HealthResponse = { status: 'error', message: 'offline' }
const defaultCheckHealth = async (): Promise<HealthResponse> => healthResponse
const checkHealth = mock(defaultCheckHealth)

let stateModule: StateModule
// mock.module rewires the module registry for the whole bun test process, so
// the real exports are captured up front and restored in afterAll — otherwise
// later test files importing '@/api/lightrag' would see the stub.
let realApiModule: Record<string, unknown>
let previousLocalStorage: PropertyDescriptor | undefined
let previousSessionStorage: PropertyDescriptor | undefined

beforeAll(async () => {
  previousLocalStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  previousSessionStorage = Object.getOwnPropertyDescriptor(globalThis, 'sessionStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    value: storageMock(),
    configurable: true
  })
  Object.defineProperty(globalThis, 'sessionStorage', {
    value: storageMock(),
    configurable: true
  })
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({ ...realApiModule, checkHealth }))

  stateModule = await import('./state')
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
  stateModule.useBackendState.setState({
    health: true,
    message: null,
    messageTitle: null,
    status: null,
    apiDocsCapability: 'unknown'
  })
  stateModule.useAuthStore.setState({
    isAuthenticated: false,
    isGuestMode: false,
    username: null,
    webuiTitle: null,
    webuiDescription: null,
    lastTokenRenewal: null,
    tokenExpiresAt: null
  })
  if (previousLocalStorage) {
    Object.defineProperty(globalThis, 'localStorage', previousLocalStorage)
  }
  if (previousSessionStorage) {
    Object.defineProperty(globalThis, 'sessionStorage', previousSessionStorage)
  }
})

const healthy = (
  extra: { api_docs_available?: boolean } = {}
): { status: 'healthy'; pipeline_busy: boolean; api_docs_available?: boolean } => ({
  status: 'healthy',
  pipeline_busy: false,
  ...extra
})

describe('apiDocsCapability state transitions', () => {
  // One sequential walk through the machine: the retention semantics only
  // show up across consecutive checks, so the steps share store state on
  // purpose (RFC #3671).
  test('unknown → retained on failure → set only by successful responses', async () => {
    const backend = () => stateModule.useBackendState.getState()
    stateModule.useBackendState.setState({
      apiDocsCapability: 'unknown',
      health: true,
      message: null,
      messageTitle: null,
      status: null
    })

    // Initial state: no successful response yet — the docs entry point must
    // stay hidden rather than guessed.
    expect(backend().apiDocsCapability).toBe('unknown')

    // A failed check before any success is NOT an old backend: stay unknown.
    healthResponse = { status: 'error', message: 'connection refused' }
    expect(await backend().check()).toBe(false)
    expect(backend().apiDocsCapability).toBe('unknown')
    expect(backend().status).toBeNull()

    // A successful response with the field false marks docs unavailable.
    healthResponse = healthy({ api_docs_available: false })
    expect(await backend().check()).toBe(true)
    expect(backend().apiDocsCapability).toBe('unavailable')

    // A later failure resets the transient status but must retain the last
    // known capability — a blip must not resurface a disabled entry point.
    healthResponse = { status: 'error', message: 'timeout' }
    expect(await backend().check()).toBe(false)
    expect(backend().status).toBeNull()
    expect(backend().apiDocsCapability).toBe('unavailable')

    // Re-enabled backend: the field flips the capability back.
    healthResponse = healthy({ api_docs_available: true })
    expect(await backend().check()).toBe(true)
    expect(backend().apiDocsCapability).toBe('available')

    // Field absent on a successful response = older backend, which always
    // serves /docs. Only valid here, after a real response.
    healthResponse = healthy()
    expect(await backend().check()).toBe(true)
    expect(backend().apiDocsCapability).toBe('available')
  })
})

describe('probeApiDocsCapability isolation', () => {
  // The probe runs when periodic health checks are disabled. Routing it
  // through check() made a single failure latch health:false, which stops the
  // document list polling for good (no timer left to restart it) and re-opens
  // the API key alert on every dismissal (RFC #3671).
  test('a failed probe mutates nothing', async () => {
    const backend = () => stateModule.useBackendState.getState()
    stateModule.useBackendState.setState({
      apiDocsCapability: 'unknown',
      health: true,
      message: null,
      messageTitle: null,
      status: null
    })

    healthResponse = { status: 'error', message: 'connection refused' }
    await backend().probeApiDocsCapability()

    expect(backend().health).toBe(true)
    expect(backend().message).toBeNull()
    expect(backend().messageTitle).toBeNull()
    expect(backend().apiDocsCapability).toBe('unknown')
  })

  test('a successful probe sets only the capability', async () => {
    const backend = () => stateModule.useBackendState.getState()
    stateModule.useBackendState.setState({
      apiDocsCapability: 'unknown',
      status: null,
      pipelineBusy: true
    })

    healthResponse = healthy({ api_docs_available: false })
    await backend().probeApiDocsCapability()

    expect(backend().apiDocsCapability).toBe('unavailable')
    // Transient health state stays owned by check() alone.
    expect(backend().status).toBeNull()
    expect(backend().pipelineBusy).toBe(true)
  })

  test('the probe no-ops once the capability is resolved', async () => {
    const backend = () => stateModule.useBackendState.getState()
    stateModule.useBackendState.setState({ apiDocsCapability: 'unavailable' })

    healthResponse = healthy({ api_docs_available: true })
    await backend().probeApiDocsCapability()

    expect(backend().apiDocsCapability).toBe('unavailable')
  })
})

describe('useBackendState.check() generation guard', () => {
  // Last-caller-wins on useBackendState.check(): concurrent health checks (a
  // credential probe racing a save-triggered re-probe, or overlapping periodic
  // checks) must not let COMPLETION ORDER decide the shared state. The pinned
  // race: an older check carrying the previous API key's failure finishes
  // AFTER the newest check already validated the replacement — its result must
  // be discarded, not written over health/message (where ApiKeyAlert's watch
  // would wrongly reopen the dialog).
  afterEach(() => {
    checkHealth.mockReset()
    checkHealth.mockImplementation(defaultCheckHealth)
    stateModule.useBackendState.setState({
      health: true,
      message: null,
      messageTitle: null,
      status: null,
      apiDocsCapability: 'unknown'
    })
  })

  test('a STALE check finishing last cannot overwrite the newest result', async () => {
    const staleResponse = deferred<HealthResponse>()
    const newestResponse = deferred<HealthResponse>()
    checkHealth
      .mockImplementationOnce(() => staleResponse.promise)
      .mockImplementationOnce(() => newestResponse.promise)

    // Old probe (previous, invalid API key) goes out first…
    const staleCheck = stateModule.useBackendState.getState().check()
    // …then the save-triggered probe for the VALID replacement key.
    const newestCheck = stateModule.useBackendState.getState().check()

    // The newest check completes first: healthy.
    newestResponse.resolve({ status: 'healthy' } as LightragStatus)
    await newestCheck
    expect(stateModule.useBackendState.getState().health).toBe(true)
    expect(stateModule.useBackendState.getState().message).toBeNull()

    // The stale check completes LAST with the old key's failure — discarded.
    staleResponse.resolve({ status: 'error', message: 'Invalid API Key' })
    const staleResult = await staleCheck
    expect(staleResult).toBe(true) // reports the state the newest check set
    expect(stateModule.useBackendState.getState().health).toBe(true)
    expect(stateModule.useBackendState.getState().message).toBeNull()
  })

  test('a single (newest) failing check still writes its result', async () => {
    const response = deferred<HealthResponse>()
    checkHealth.mockImplementationOnce(() => response.promise)

    const check = stateModule.useBackendState.getState().check()
    response.resolve({ status: 'error', message: 'Invalid API Key' })

    expect(await check).toBe(false)
    expect(stateModule.useBackendState.getState().health).toBe(false)
    expect(stateModule.useBackendState.getState().message).toBe('Invalid API Key')
  })
})

describe('tab title follows the auth store', () => {
  // The server stamps the title into the entry HTML for the first paint, but
  // that value is frozen at page load: `/health` (via `setCustomTitle`) and the
  // login response are what carry a restarted server's new title into an open
  // tab, and under `bun run dev` they are the ONLY source. This pins that the
  // store actually drives `document.title` — the subscription is registered at
  // module import, so a regression here is silent in every other test.

  // Cast through `unknown`: the DOM lib types `globalThis.document`/`window` as
  // required and fully-featured, which neither a stub nor `delete` can satisfy.
  const g = globalThis as unknown as {
    document?: { title: string }
    window?: { __LIGHTRAG_CONFIG__?: { webuiTitle?: string | null } }
  }

  // `window` is installed only for the duration of the call, then the previous
  // value is PUT BACK rather than deleted (deleting would strip the preloaded
  // DOM for every later test file).
  const withInjectedTitle = (injected: string, body: () => void): void => {
    const previous = g.window
    g.window = { __LIGHTRAG_CONFIG__: { webuiTitle: injected } }
    try {
      body()
    } finally {
      g.window = previous
    }
  }

  let realDocument: typeof g.document
  const token = jwt({ sub: 'alice', exp: Math.floor(Date.now() / 1000) + 3600 })

  beforeEach(() => {
    // `document` is read at call time, so installing it here works whether or
    // not another test file imported the store first. The real (preloaded) one
    // is kept so `afterEach` restores it for later tests and files.
    realDocument = g.document
    g.document = { title: 'LightRAG' }
  })

  afterEach(() => {
    g.document = realDocument
    localStorage.clear()
  })

  test('health-driven title updates, and reverts to the default when unset', () => {
    const auth = () => stateModule.useAuthStore.getState()

    auth().setCustomTitle('My Graph KB', 'Simple and Fast Graph Based RAG System')
    expect(g.document!.title).toBe('My Graph KB')

    // A server restarted with a different WEBUI_TITLE reaches an open tab on
    // the next poll.
    auth().setCustomTitle('Renamed KB', null)
    expect(g.document!.title).toBe('Renamed KB')

    // Cleared on the server: back to the generic name, not the stale one.
    auth().setCustomTitle(null, null)
    expect(g.document!.title).toBe('LightRAG')
  })

  test('a cleared title is not overridden by the injected page-load value', () => {
    // Regression (Codex review on PR #3763): the page was served while the
    // deployment had a title, so that title is frozen into the runtime config
    // in <head>. The server then restarts with WEBUI_TITLE unset and /health
    // reports null. The store is authoritative — the tab must revert to the
    // default like the header does, not resurrect the injected snapshot and
    // stay wrong until a reload.
    const auth = () => stateModule.useAuthStore.getState()

    withInjectedTitle('Old KB', () => {
      auth().setCustomTitle('Old KB', null)
      expect(g.document!.title).toBe('Old KB')

      auth().setCustomTitle(null, null)
      expect(g.document!.title).toBe('LightRAG')
    })
  })

  test('login applies the title, and logout keeps it', () => {
    const auth = () => stateModule.useAuthStore.getState()

    auth().login(token, false, '1.0.0', '1.0.0', 'My Graph KB', null)
    expect(g.document!.title).toBe('My Graph KB')

    // The title is deployment-level, not per-session: the login page belongs
    // to the same deployment, so signing out must not rename the tab.
    auth().logout()
    expect(g.document!.title).toBe('My Graph KB')
  })
})

describe('initAuthState local validation', () => {
  // Local token validity (client-state partitioning doc §5). These cases pin
  // the REPLACEMENT of the old "token exists ⇒ authenticated" rule — an
  // implementation that only checks existence fails every rejection case. They
  // go through `initAuthState`, the one caller of `isTokenLocallyValid`, so each
  // also pins what a rejected token leaves behind in storage.
  const futureExp = Math.floor(Date.now() / 1000) + 3600
  const pastExp = Math.floor(Date.now() / 1000) - 3600

  beforeEach(() => {
    localStorage.clear()
  })

  const seed = (token: string) => {
    localStorage.setItem('LIGHTRAG-API-TOKEN', token)
    localStorage.setItem('LIGHTRAG-LAST-TOKEN-RENEWAL', '2026-01-01 00:00:00 (UTC+0)')
  }

  test.each([
    ['not three segments', 'broken-token'],
    ['payload not Base64URL', 'head.@@@@@.sig'],
    ['payload not JSON', `head.${b64url('nope')}.sig`],
    ['missing exp', jwt({ sub: 'alice' })],
    ['expired exp', jwt({ sub: 'alice', exp: pastExp })]
  ])('%s ⇒ unauthenticated AND token + companion keys cleared', (_label, token) => {
    seed(token)
    const state = stateModule.initAuthState()
    expect(state.isAuthenticated).toBe(false)
    expect(state.lastTokenRenewal).toBeNull()
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).toBeNull()
    expect(localStorage.getItem('LIGHTRAG-LAST-TOKEN-RENEWAL')).toBeNull()
  })

  test('valid unexpired token ⇒ authenticated, nothing cleared', () => {
    seed(jwt({ sub: 'alice', role: 'user', exp: futureExp }))
    const state = stateModule.initAuthState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.username).toBe('alice')
    expect(localStorage.getItem('LIGHTRAG-API-TOKEN')).not.toBeNull()
    expect(localStorage.getItem('LIGHTRAG-LAST-TOKEN-RENEWAL')).not.toBeNull()
  })

  test('a base64url payload yields the CLAIMS, not a blank authenticated session', () => {
    // Regression: validation used the url-safe decoder while the claims
    // parser used bare `atob`, so the two disagreed about the SAME token —
    // it was admitted, then read as `{}`. The session then ran with no
    // username, and `navigationService` records LIGHTRAG-PREVIOUS-USER only
    // `if (currentUsername)`: that user's logout wrote no identity marker at
    // all, so the retrieval-history cleanup never fired for them and their
    // conversations were left to whoever logged in next.
    //
    // Whether a payload needs the url alphabet depends on its exact bytes,
    // `exp` included, so it varies per issuance for the SAME user — which is
    // what made this intermittent. Hence the fixed vector below.
    //
    // The sub is ASCII on purpose: `atob` yields one character per BYTE, so
    // a non-ASCII sub comes back mojibake even once it decodes. That is a
    // separate defect from this one and is not what this test pins.
    const token = jwt({ sub: '~~h', role: 'user', exp: 9999999999 })
    const segment = token.split('.')[1]
    // Only meaningful if the payload really needs the url alphabet.
    expect(segment).toMatch(/[-_]/)
    expect(() => atob(segment)).toThrow()

    seed(token)
    const state = stateModule.initAuthState()

    expect(state.isAuthenticated).toBe(true)
    expect(state.username).toBe('~~h')
    expect(state.isGuestMode).toBe(false)
    expect(state.tokenExpiresAt).toBe(9999999999 * 1000)
  })

  test('a locally-valid guest token counts as a valid login (welcome page is skipped by it)', () => {
    seed(jwt({ sub: 'guest', role: 'guest', exp: futureExp }))
    const state = stateModule.initAuthState()
    expect(state.isAuthenticated).toBe(true)
    expect(state.isGuestMode).toBe(true)
  })
})
