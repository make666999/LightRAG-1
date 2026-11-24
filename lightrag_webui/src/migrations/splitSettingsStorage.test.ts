import { afterAll, afterEach, beforeEach, describe, expect, test } from 'bun:test'
import {
  runSettingsStorageSplitMigration,
  getSettingsMigrationError,
  hasFutureSettingsEnvelope,
  resetSettingsMigrationErrorForTests,
  wasHistoryDroppedDuringMigration,
  acceptSettingsDataLoss,
  isSettingsDataLostError,
  SETTINGS_STORAGE_VERSION_AFTER_SPLIT,
  QUERY_SETTINGS_STORE_VERSION,
  RETRIEVAL_HISTORY_STORE_VERSION
} from './splitSettingsStorage'
import { LEGACY_SETTINGS_VERSION_CAP } from './legacySettingsChain'
import {
  LEGACY_SETTINGS_STORAGE_KEY,
  QUERY_SETTINGS_STORAGE_KEY,
  WEBUI_RETRIEVAL_HISTORY_KEY,
  WORKSPACE_RETRIEVAL_HISTORY_KEY
} from '@/lib/storageKeys'
import type { RetrievalHistoryStore } from '@/stores/retrievalHistory'

/** In-memory Storage double; `failAfterWrites` / `failValueLongerThan` inject faults. */
class FakeStorage implements Storage {
  data = new Map<string, string>()
  writeCount = 0
  failAfterWrites: number | null = null
  // Quota-realistic mode: values LARGER than this fail, small ones succeed
  // (a real quota keeps accepting small writes after rejecting a big one).
  failValueLongerThan: number | null = null
  // 'Error' simulates a generic crash; 'QuotaExceededError' the quota path.
  failureName = 'Error'

  get length() {
    return this.data.size
  }
  clear() {
    this.data.clear()
  }
  getItem(key: string) {
    return this.data.get(key) ?? null
  }
  key(index: number) {
    return [...this.data.keys()][index] ?? null
  }
  removeItem(key: string) {
    this.data.delete(key)
  }
  setItem(key: string, value: string) {
    const overQuota =
      this.failValueLongerThan !== null && value.length > this.failValueLongerThan
    const crashed =
      this.failAfterWrites !== null && this.writeCount >= this.failAfterWrites
    if (overQuota || crashed) {
      const error = new Error('write failed (injected)')
      error.name = this.failureName
      throw error
    }
    this.writeCount += 1
    this.data.set(key, value)
  }
}

/**
 * Total-byte quota — the model in which a large write can SUCCEED and a
 * LATER write then fails because the budget is exhausted. `freeOnReplace`
 * covers both real browser behaviors: some free a replaced key's old bytes
 * before checking, some check against the current total. Optional
 * `collapseAfterWrites` / `collapsedBudget` models a budget that collapses
 * partway through a run (e.g. another tab consuming space while the split runs).
 */
class ByteQuotaStorage implements Storage {
  data = new Map<string, string>()
  removed: string[] = []
  private writes = 0
  constructor(
    public budget: number,
    private freeOnReplace = true,
    private collapseAfterWrites?: number,
    private collapsedBudget?: number
  ) {}

  get length() {
    return this.data.size
  }
  private used(exceptKey?: string) {
    let total = 0
    for (const [key, value] of this.data) {
      if (key !== exceptKey) total += key.length + value.length
    }
    return total
  }
  clear() {
    this.data.clear()
  }
  getItem(key: string) {
    return this.data.get(key) ?? null
  }
  key(index: number) {
    return [...this.data.keys()][index] ?? null
  }
  removeItem(key: string) {
    this.removed.push(key)
    this.data.delete(key)
  }
  setItem(key: string, value: string) {
    const used = this.freeOnReplace ? this.used(key) : this.used()
    if (used + key.length + value.length > this.budget) {
      const error = new Error('quota exceeded (byte budget)')
      error.name = 'QuotaExceededError'
      throw error
    }
    this.data.set(key, value)
    if (
      this.collapseAfterWrites !== undefined &&
      this.collapsedBudget !== undefined &&
      ++this.writes === this.collapseAfterWrites
    ) {
      this.budget = this.collapsedBudget
    }
  }
}

const HISTORY = [
  { id: 'h1', role: 'user', content: 'admin debug question' },
  { id: 'h2', role: 'assistant', content: 'answer' }
]

const legacyEnvelope = (version: number, state: Record<string, unknown>) =>
  JSON.stringify({ state, version })

const v21State = () => ({
  theme: 'dark',
  language: 'zh',
  apiKey: 'secret',
  querySettings: { mode: 'hybrid', top_k: 7, history_turns: 0 },
  retrievalHistory: HISTORY
})

/** Envelope-level loss marker, by literal name (it is module-private). */
const SETTINGS_LOST_MARKER_KEY = 'lightragSettingsLostToQuota'

const parsedFrom = (s: Storage, key: string) => JSON.parse(s.getItem(key)!)

let storage: FakeStorage

beforeEach(() => {
  storage = new FakeStorage()
  resetSettingsMigrationErrorForTests()
})

const parsed = (key: string) => parsedFrom(storage, key)

describe('field mapping (fixed and exhaustive)', () => {
  test('querySettings → query-settings key, retrievalHistory → webui history, workspace history = []', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    // The admin history is copied verbatim…
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
    // …and the workspace history has NO legacy source: it starts EMPTY.
    // Copying the admin debug record here is exactly the defect the split
    // removes (it would even feed bypass-mode conversation_history).
    expect(parsed(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])

    // Legacy envelope: migrated fields stripped, version bumped, rest kept.
    const legacy = parsed(LEGACY_SETTINGS_STORAGE_KEY)
    expect(legacy.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(legacy.state.querySettings).toBeUndefined()
    expect(legacy.state.retrievalHistory).toBeUndefined()
    expect(legacy.state.apiKey).toBe('secret')
    expect(legacy.state.theme).toBe('dark')
  })

  test('no legacy envelope → nothing written', () => {
    runSettingsStorageSplitMigration(storage)
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    expect(getSettingsMigrationError()).toBeNull()
  })

  test('corrupt envelope treated as no legacy data', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, '{not json')
    runSettingsStorageSplitMigration(storage)
    expect(getSettingsMigrationError()).toBeNull()
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    // Broken blob left in place.
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe('{not json')
  })
})

describe('version chain reuse (normalize before split)', () => {
  // Expected outcomes prove the REUSED chain ran with its version guards:
  // v1/v6 pass the v15 step (mode forced to 'mix') and the v12 step (history
  // cleared: pre-MessageWithError format); a v20 envelope passes neither, so
  // its own mode and history survive intact.
  test.each([
    [1, 'mix', []],
    [6, 'mix', []],
    [20, 'local', HISTORY]
  ] as const)(
    'v%i envelopes go through the legacy chain, not "no data"',
    (version, expectedMode, expectedHistory) => {
      const state: Record<string, unknown> = {
        theme: 'light',
        querySettings: { mode: 'local', history_turns: 0 },
        retrievalHistory: HISTORY
      }
      storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(version, state))
      runSettingsStorageSplitMigration(storage)

      expect(getSettingsMigrationError()).toBeNull()
      const qs = parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings
      expect(qs.mode).toBe(expectedMode)
      expect(qs.history_turns).toBe(0)
      expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(
        expectedHistory as unknown[]
      )
    }
  )

  test('version > cap: nothing read, cleaned or overwritten', () => {
    const envelope = legacyEnvelope(99, v21State())
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, envelope)
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(envelope)
  })

  test('our own post-split version (22) is a no-op', () => {
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(SETTINGS_STORAGE_VERSION_AFTER_SPLIT, { theme: 'dark' })
    )
    runSettingsStorageSplitMigration(storage)
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
  })
})

describe('new keys always win', () => {
  test('existing new keys are never overwritten by legacy values', () => {
    const userEdited = { state: { querySettings: { mode: 'naive' } }, version: 1 }
    storage.setItem(QUERY_SETTINGS_STORAGE_KEY, JSON.stringify(userEdited))
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))

    runSettingsStorageSplitMigration(storage)

    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('naive')
    // The other targets were still filled, and legacy was cleaned.
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.querySettings).toBeUndefined()
  })

  test('triple execution: runs 2 and 3 are no-ops and do not clobber user edits', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)

    // User edits the new key after migration.
    storage.setItem(
      QUERY_SETTINGS_STORAGE_KEY,
      JSON.stringify({ state: { querySettings: { mode: 'global' } }, version: 1 })
    )
    const snapshotAfterEdit = new Map(storage.data)

    runSettingsStorageSplitMigration(storage)
    runSettingsStorageSplitMigration(storage)

    expect(storage.data).toEqual(snapshotAfterEdit)
  })

  test('old-tab rewrite: a full v21 envelope written back later is cleaned, never merged', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)

    // A tab still running pre-split code rewrites the whole legacy envelope
    // with DIFFERENT settings and history.
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, {
        ...v21State(),
        querySettings: { mode: 'naive', top_k: 1, history_turns: 0 },
        retrievalHistory: [{ id: 'x', role: 'user', content: 'from old tab' }]
      })
    )

    runSettingsStorageSplitMigration(storage)

    // Cleaned…
    const legacy = parsed(LEGACY_SETTINGS_STORAGE_KEY)
    expect(legacy.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(legacy.state.querySettings).toBeUndefined()
    expect(legacy.state.retrievalHistory).toBeUndefined()
    // …but the new keys are unchanged: the old tab's post-migration edits
    // are explicitly allowed to be lost (this is a cleanup, not an absorb).
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
  })
})

describe('crash and partial-failure behavior', () => {
  test('write failure mid-copy: legacy untouched, error recorded, retry converges', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    storage.failAfterWrites = 2 // legacy write counted as 1; allow one target write

    runSettingsStorageSplitMigration(storage)

    // Physically partial — that is allowed…
    expect(getSettingsMigrationError()).not.toBeNull()
    // …but the legacy envelope keeps ALL its fields (cleanup never ran).
    const legacy = parsed(LEGACY_SETTINGS_STORAGE_KEY)
    expect(legacy.version).toBe(21)
    expect(legacy.state.querySettings).toBeDefined()
    expect(legacy.state.retrievalHistory).toBeDefined()
    // The interrupted run wrote NO defaults into not-yet-migrated keys:
    // whatever keys exist carry legacy-derived values, and at least one
    // target is still missing.
    const missing = [
      QUERY_SETTINGS_STORAGE_KEY,
      WEBUI_RETRIEVAL_HISTORY_KEY,
      WORKSPACE_RETRIEVAL_HISTORY_KEY
    ].filter((key) => storage.getItem(key) == null)
    expect(missing.length).toBeGreaterThan(0)

    // Retry (next startup): converges to the same result as one clean run.
    storage.failAfterWrites = null
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
    expect(parsed(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
  })

  test('crash after copy but before cleanup: re-run cleans without overwriting', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    storage.failAfterWrites = 4 // 1 legacy + 3 targets; fail on the cleanup write

    runSettingsStorageSplitMigration(storage)
    expect(getSettingsMigrationError()).not.toBeNull()
    // All targets written, legacy still v21 with fields.
    expect(storage.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)).not.toBeNull()
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(21)

    storage.failAfterWrites = null
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
  })

  test('workspace entry may execute the one-time migration and writes ALL targets', () => {
    // The migration is entry-agnostic: whichever entry runs first writes the
    // full set (including the webui history key).
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)
    expect(storage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).not.toBeNull()
    expect(storage.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)).not.toBeNull()
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).not.toBeNull()
  })
})

describe('hasFutureSettingsEnvelope (rule 3 extended to the settings store)', () => {
  test('an envelope from a NEWER client (version > 22) is detected', () => {
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(SETTINGS_STORAGE_VERSION_AFTER_SPLIT + 1, { theme: 'dark' })
    )
    expect(hasFutureSettingsEnvelope(storage)).toBe(true)
  })

  test('current, legacy, absent and corrupt envelopes are NOT future', () => {
    const nonFuturePayloads = [
      legacyEnvelope(SETTINGS_STORAGE_VERSION_AFTER_SPLIT, { theme: 'dark' }),
      legacyEnvelope(21, v21State()),
      JSON.stringify({ state: {}, version: 'later' }),
      '{not json'
    ]
    for (const raw of nonFuturePayloads) {
      storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, raw)
      expect(hasFutureSettingsEnvelope(storage)).toBe(false)
    }
    storage.removeItem(LEGACY_SETTINGS_STORAGE_KEY)
    expect(hasFutureSettingsEnvelope(storage)).toBe(false)
    expect(hasFutureSettingsEnvelope(null)).toBe(false)
  })
})

describe('legacy language choice synthesis (languageUserSelected)', () => {
  // Legacy `language` only ever changed through the selector, so a persisted
  // NON-DEFAULT value proves an explicit choice; the marker is synthesized
  // during the split so an upgrade cannot let the browser language silently
  // replace it. A persisted 'en' is ambiguous and stays non-explicit.
  test('a non-default legacy language is marked as an explicit choice', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State())) // language: 'zh'
    runSettingsStorageSplitMigration(storage)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.languageUserSelected).toBe(true)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.language).toBe('zh')
  })

  test('the default \'en\' (or a missing language) stays non-explicit', () => {
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, { ...v21State(), language: 'en' })
    )
    runSettingsStorageSplitMigration(storage)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.languageUserSelected).toBe(false)

    const withoutLanguage: Record<string, unknown> = { ...v21State() }
    delete withoutLanguage.language
    storage = new FakeStorage()
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, withoutLanguage))
    runSettingsStorageSplitMigration(storage)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.languageUserSelected).toBe(false)
  })
})

describe('quota-aware degradation (history is non-critical test data)', () => {
  // Product decision: a legacy history near the browser quota cannot coexist
  // with its migrated copy. Rather than locking the user into a migration
  // error whose retry can never succeed, the split retries WITHOUT the
  // history copy, records the drop (surfaced as a one-time toast by the
  // entry shells), and everything else migrates normally.
  test('QuotaExceededError on the history copy drops the history and completes', () => {
    // A history too large to duplicate: the quota rejects its serialized
    // copy while every other (small) write keeps succeeding — which is what
    // lets the degraded retry converge WITHOUT any quota being freed.
    const bigHistory = [{ id: 'h1', role: 'user', content: 'x'.repeat(4000) }]
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, { ...v21State(), retrievalHistory: bigHistory })
    )
    storage.failValueLongerThan = 2000
    storage.failureName = 'QuotaExceededError'

    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(true)

    // Everything except the history migrated normally…
    expect(parsed(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).state.retrievalHistory).toBeUndefined()
    // …and the history keys hold EMPTY histories, not the oversized copy.
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(parsed(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
  })

  test('a GENERIC write failure keeps the original fail-and-retry semantics', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    storage.failAfterWrites = 2
    storage.failureName = 'Error'

    runSettingsStorageSplitMigration(storage)

    expect(getSettingsMigrationError()).not.toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(parsed(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(21)
  })

  test('a clean run reports no drop', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))
    runSettingsStorageSplitMigration(storage)
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(parsed(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(HISTORY)
  })
})

describe('quota degradation under a byte budget', () => {
  const bigHistory = [{ id: 'h1', role: 'user', content: 'x'.repeat(1200) }]
  const legacyWithBigHistory = (overrides: Record<string, unknown> = {}) =>
    legacyEnvelope(21, { ...v21State(), retrievalHistory: bigHistory, ...overrides })

  const seedTightStore = (
    budget: number,
    {
      stateOverrides = {},
      unrelatedBytes = 1200,
      freeOnReplace = true,
      extraKeys = {} as Record<string, string>
    }: {
      stateOverrides?: Record<string, unknown>
      unrelatedBytes?: number
      freeOnReplace?: boolean
      extraKeys?: Record<string, string>
    } = {}
  ) => {
    const quotaStorage = new ByteQuotaStorage(100000, freeOnReplace)
    for (const [key, value] of Object.entries(extraKeys)) quotaStorage.setItem(key, value)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory(stateOverrides))
    if (unrelatedBytes > 0) quotaStorage.setItem('unrelated::blob', 'y'.repeat(unrelatedBytes))
    quotaStorage.budget = budget
    return quotaStorage
  }

  const restoreCapacityAndRerun = (quotaStorage: ByteQuotaStorage, deleteBlob = true) => {
    if (deleteBlob) quotaStorage.data.delete('unrelated::blob')
    quotaStorage.budget = 100000
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(quotaStorage)
  }

  const expectCompletedWithEmptyHistory = (quotaStorage: ByteQuotaStorage) => {
    const read = (key: string) => parsedFrom(quotaStorage, key)
    // No dead end: the split completed under the degradation policy…
    expect(getSettingsMigrationError()).toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(true)
    // …every target exists, with the history dropped to empty…
    expect(read(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(read(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(read(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    // …and the legacy envelope was cleaned and bumped, so the next reload is
    // a no-op instead of repeating the failure forever.
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).state.retrievalHistory).toBeUndefined()
  }

  test('the history copy itself does not fit: the retry writes an empty one', () => {
    const quotaStorage = new ByteQuotaStorage(2100)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    expectCompletedWithEmptyHistory(quotaStorage)
    // Nothing to reclaim — the oversized write never landed.
    expect(quotaStorage.removed).toEqual([])
  })

  test('the copy LANDS and the cleanup hits the quota: the history is KEPT', () => {
    // Regression: this used to reclaim the migrated copy — the one piece of
    // data the split exists to carry across — and rewrite it EMPTY, even
    // though it had provably fit. The copy is phase 1's last target and
    // `writeIfAbsent` records a key only once its write returned, so a
    // failure with the copy present pins the blame on the ONE write after
    // it: the legacy-envelope cleanup. That payload is SMALLER than what it
    // replaces (it sheds querySettings and the history), so the rejection is
    // an engine that cannot shrink a key in place, not a shortage — removing
    // the key first completes the split with nothing lost.
    //
    // Budget calibrated so the history copy still fits (total 2908 B) while
    // the cleanup write that follows it does not (needs 3025 B); after the
    // remove it needs only 1614 B.
    const quotaStorage = new ByteQuotaStorage(2950, false)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    const read = (key: string) => parsedFrom(quotaStorage, key)
    // The legacy envelope is the key being rewritten, so it is the one that
    // gets removed; the migrated copy must NOT be (the scratch key of the
    // headroom probe is bookkeeping, so assert the intent, not the list)…
    expect(quotaStorage.removed).toContain(LEGACY_SETTINGS_STORAGE_KEY)
    expect(quotaStorage.removed).not.toContain(WEBUI_RETRIEVAL_HISTORY_KEY)
    // …the migrated history survived in full, and no degradation was claimed…
    expect(read(WEBUI_RETRIEVAL_HISTORY_KEY).state.history).toEqual(bigHistory)
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(getSettingsMigrationError()).toBeNull()
    // …and the split still completed: every target exists and the legacy
    // envelope is cleaned and bumped, so the next reload is a no-op.
    expect(read(QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe('hybrid')
    expect(read(WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toEqual([])
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).version).toBe(
      SETTINGS_STORAGE_VERSION_AFTER_SPLIT
    )
    expect(read(LEGACY_SETTINGS_STORAGE_KEY).state.retrievalHistory).toBeUndefined()
  })

  test('the cleanup stays impossible even after the remove: the raw envelope returns', () => {
    // The other side of the shrink-safe write: when removing the key does
    // NOT make room either (another origin ate the budget between the two
    // statements), the exact bytes just removed go back. A run that left NO
    // legacy envelope would let the NEXT run read "no legacy data" and
    // declare the split done over fields it never migrated.
    const quotaStorage = new ByteQuotaStorage(2950, false)
    const raw = legacyWithBigHistory()
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, raw)
    const realSetItem = quotaStorage.setItem.bind(quotaStorage)
    // Reject every legacy-envelope write, including the post-remove retry;
    // the restore of the original bytes must still be allowed through.
    quotaStorage.setItem = (key: string, value: string) => {
      if (key === LEGACY_SETTINGS_STORAGE_KEY && value !== raw) {
        const error = new Error('quota exceeded (byte budget)')
        error.name = 'QuotaExceededError'
        throw error
      }
      realSetItem(key, value)
    }

    runSettingsStorageSplitMigration(quotaStorage)

    // Rules 6/7: the failure is recorded so dependent stores refuse to
    // hydrate and the entry offers a retry…
    expect(getSettingsMigrationError()).not.toBeNull()
    // …and the legacy envelope is back, byte for byte, rather than absent.
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(raw)
  })

  test('the FIRST target write already fails: the legacy history is freed', () => {
    // The dead end this covers: when the legacy envelope alone leaves less
    // room than the small query-settings target needs, nothing of ours ever
    // lands, so there is nothing to reclaim and the degraded retry repeats
    // the identical failing write — on this run and on every reload, pinning
    // the user on MigrationErrorScreen with a retry that can never succeed.
    // The only bytes left to free are the legacy history's own, and by then
    // the degradation has already decided to discard it.
    const quotaStorage = new ByteQuotaStorage(1450, false)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    // Nothing of ours had landed, so nothing of ours was reclaimed…
    expect(quotaStorage.removed).not.toContain(WEBUI_RETRIEVAL_HISTORY_KEY)
    // …and the split still completed under the degradation policy.
    expectCompletedWithEmptyHistory(quotaStorage)
  })

  test('the same case on an engine that CAN shrink a key in place', () => {
    // freeOnReplace: the legacy envelope is rewritten smaller directly, so
    // the key is never removed and no crash window exists at all.
    const quotaStorage = new ByteQuotaStorage(1450)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    expect(quotaStorage.removed).toEqual([])
    expectCompletedWithEmptyHistory(quotaStorage)
  })

  test('a quota that stays exhausted is reported, not silently declared done', () => {
    // Freeing the legacy history cannot always be enough — here another key
    // holds the store. The run must record the failure so dependent stores
    // refuse to hydrate and the entry offers a retry (rules 6/7), and it must
    // leave a legacy envelope behind: a run that found NO envelope would
    // declare the split done over data it never migrated.
    const quotaStorage = seedTightStore(1250, { unrelatedBytes: 1000, freeOnReplace: false })

    runSettingsStorageSplitMigration(quotaStorage)

    expect(getSettingsMigrationError()).not.toBeNull()
    expect(wasHistoryDroppedDuringMigration()).toBe(false)
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).not.toBeNull()
  })

  test('a store with room to complete is NOT dead-ended by caution', () => {
    // The other side of the ladder, and the reason it is not gated on a
    // probe write. A probe can only certify SLACK, while the removal needs
    // something weaker — that the store is over quota by no more than the
    // removal frees, which here is the whole legacy history. Gating on one
    // therefore refuses the very cases the removal exists to rescue: this
    // store has room to finish, and a gated version reports a failure whose
    // retry can never succeed, pinning the user on MigrationErrorScreen.
    //
    // freeOnReplace: false is the engine that cannot shrink a key in place,
    // i.e. the one whose first write fails for reasons that have nothing to
    // do with how full the store is.
    const quotaStorage = seedTightStore(1300, { unrelatedBytes: 0, freeOnReplace: false })

    runSettingsStorageSplitMigration(quotaStorage)

    expectCompletedWithEmptyHistory(quotaStorage)
  })

  test('an over-quota store keeps a MIGRATABLE envelope rather than none', () => {
    // Regression. The rewrite path removes the key first so an engine that
    // cannot replace a value in place can still shrink it, and the payload
    // is never larger than what it replaces — so the removal can never be
    // what pushes a store over its quota. It CAN still fail on a store
    // already over quota, and there the bytes just removed are larger still,
    // so the faithful restore fails too. What must not happen is the key
    // ending up ABSENT: the next run would read "no legacy data", return
    // early with NO error, and the app would start on defaults having
    // migrated nothing — a silent total loss instead of a visible failure.
    //
    // Hence the write-back ladder's last rung: the minimal envelope fits in
    // a window where the real bytes do not.
    const quotaStorage = seedTightStore(1300)

    runSettingsStorageSplitMigration(quotaStorage)

    // Nothing of substance could be written, so the run reports rather than
    // pretending it succeeded…
    expect(getSettingsMigrationError()).not.toBeNull()
    // …and what is left is still an envelope the chain will migrate, not a
    // browser that looks like it never had one.
    const left = quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    expect(left).not.toBeNull()
    expect(JSON.parse(left!).version).toBeLessThanOrEqual(LEGACY_SETTINGS_VERSION_CAP)

    // The reload keeps finding legacy data to work on, and keeps reporting.
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(quotaStorage)
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).not.toBeNull()
  })

  test('a destroyed envelope is REFUSED, not completed, when capacity returns', () => {
    // The defect codex reported. The last rung used to be a bare
    // `{state:{}, version:21}` — a perfectly VALID envelope. Once the user
    // freed space and pressed the error screen's own Reload, the next run
    // migrated its empty state successfully, stamped v22 and never retried:
    // theme, language and apiKey gone, app starting clean, no error anywhere.
    // The retry button converted a visible failure into silent data loss.
    const quotaStorage = seedTightStore(1300)

    runSettingsStorageSplitMigration(quotaStorage)
    // Reported as unrecoverable from the FIRST screen, not as "try again".
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)
    const tombstone = quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)
    expect(JSON.parse(tombstone!).state).toEqual({})

    // Capacity returns and the user reloads — the sequence that used to lose
    // the settings for good.
    restoreCapacityAndRerun(quotaStorage)

    // Still refused, still as unrecoverable, and NOTHING written: no targets
    // invented from an empty state, no v22 stamp over data never read.
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(tombstone)
    expect(quotaStorage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
  })

  test('accepting the loss is what clears it, and only then', () => {
    const quotaStorage = seedTightStore(1300)
    runSettingsStorageSplitMigration(quotaStorage)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)

    // Only a person gets past it; nothing does this on its own.
    quotaStorage.budget = 100000
    acceptSettingsDataLoss(quotaStorage)
    expect(getSettingsMigrationError()).toBeNull()
    // Nothing survived, so the key goes and the reload starts from defaults.
    expect(quotaStorage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBeNull()
    runSettingsStorageSplitMigration(quotaStorage)
    expect(getSettingsMigrationError()).toBeNull()
  })

  test('scalars are shed one at a time before the envelope is emptied', () => {
    // The hole the object-only rung left. By the time the ladder reaches a
    // payload the split has already stripped of its objects, "drop the
    // non-primitives" finds nothing to drop — so without a scalar ladder the
    // next rung was the empty envelope and EVERY scalar went at once. Here
    // the largest one goes and the cheap ones survive.
    // Sized to fit the envelope minus its biggest scalar, but not the whole
    // one (see the rung ladder in reducedLegacyPayloads).
    const quotaStorage = seedTightStore(1340, { stateOverrides: { apiKey: 'k'.repeat(60) } })

    runSettingsStorageSplitMigration(quotaStorage)

    const left = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    // The oversized apiKey was shed; the cheap scalars survived.
    expect(left.state.apiKey).toBeUndefined()
    expect(left.state.theme).toBe('dark')
    expect(left.state.language).toBe('zh')
    // A scalar was genuinely lost, so it is marked and reported: an unmarked
    // partial envelope would migrate "successfully" on the next load.
    expect(left[SETTINGS_LOST_MARKER_KEY]).toBe(true)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)
  })

  test('accepting a PARTIAL loss keeps what survived', () => {
    // The counterpart to the empty case: acceptance adapts to what is left,
    // so a user who says "continue" still gets the settings that fit.
    const quotaStorage = seedTightStore(1340, { stateOverrides: { apiKey: 'k'.repeat(60) } })
    runSettingsStorageSplitMigration(quotaStorage)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)

    quotaStorage.budget = 100000
    acceptSettingsDataLoss(quotaStorage)
    // Fields remained, so the key stays (un-marked) and the reload migrates
    // them instead of starting from defaults.
    const kept = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    expect(kept[SETTINGS_LOST_MARKER_KEY]).toBeUndefined()
    expect(kept.state.theme).toBe('dark')

    restoreCapacityAndRerun(quotaStorage, false)
    expect(getSettingsMigrationError()).toBeNull()
    expect(parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY).state.theme).toBe('dark')
  })

  test('the scalar rung keeps theme/language/apiKey when the payload cannot fit', () => {
    // Between "the real bytes" and "nothing" there is a rung worth having:
    // the same envelope minus its non-primitive fields. The unbounded fields
    // are objects (querySettings, the history); the settings a user would
    // actually miss are scalars worth a few dozen bytes, and they fit in
    // windows where the payload does not.
    //
    // Regression (P1) on the SECOND half of that rung: whether shedding the
    // objects is free. This run never wrote the query-settings target — the
    // ladder is reached from `freeLegacyHistory`, which exists precisely for
    // the case where the FIRST target write already failed — so dropping
    // `querySettings` here drops the only copy that exists. The rung used to
    // be unmarked unconditionally ("objects are already in their own keys"),
    // so the reload migrated the reduced envelope successfully, the store
    // hydrated default query settings, and `acceptSettingsDataLoss` was
    // never offered for settings the browser never handed over.
    // Room for the scalars once the legacy envelope goes, but not for the
    // history-free payload that still carries querySettings.
    const quotaStorage = seedTightStore(1500, {
      stateOverrides: { querySettings: { mode: 'hybrid', top_k: 7, blob: 'q'.repeat(600) } }
    })

    runSettingsStorageSplitMigration(quotaStorage)

    // The scalars still survive the rung — that half is unchanged…
    const left = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    expect(left.state).toEqual({ theme: 'dark', language: 'zh', apiKey: 'secret' })
    // …and the query settings that went with it are nowhere else.
    expect(quotaStorage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    // So it IS a loss, said so on the envelope and on this run's error.
    expect(left[SETTINGS_LOST_MARKER_KEY]).toBe(true)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)

    // Capacity returns and the user reloads: still refused, and nothing
    // written — no query-settings target invented from a field that is gone.
    restoreCapacityAndRerun(quotaStorage)

    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)
    expect(quotaStorage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()

    // Only the person gets past it, and the scalars they still have survive.
    acceptSettingsDataLoss(quotaStorage)
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(quotaStorage)

    expect(getSettingsMigrationError()).toBeNull()
    const migrated = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    expect(migrated.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(migrated.state.theme).toBe('dark')
    expect(migrated.state.language).toBe('zh')
    expect(migrated.state.apiKey).toBe('secret')
    // The query settings really are gone — now behind an explicit acceptance
    // instead of behind nothing at all.
    expect(parsedFrom(quotaStorage, QUERY_SETTINGS_STORAGE_KEY).state).toEqual({})
  })

  test('the object rung is FREE when every dropped field already has its key', () => {
    // The other side of the same rule, and what keeps the fix from being a
    // blanket "objects are always a loss". Rule 4 makes a PRE-EXISTING
    // query-settings key authoritative — the legacy copy was never going to
    // be written anywhere — so shedding `querySettings` from the envelope
    // forfeits nothing and must not raise a loss the user has to accept.
    const quotaStorage = seedTightStore(1500, {
      extraKeys: {
        [QUERY_SETTINGS_STORAGE_KEY]: JSON.stringify({
          state: { querySettings: { mode: 'local' } },
          version: 1
        })
      },
      stateOverrides: { querySettings: { mode: 'hybrid', top_k: 7, blob: 'q'.repeat(600) } }
    })

    runSettingsStorageSplitMigration(quotaStorage)

    const left = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    expect(left.state).toEqual({ theme: 'dark', language: 'zh', apiKey: 'secret' })
    // No marker: nothing the rung dropped existed only in this envelope.
    expect(SETTINGS_LOST_MARKER_KEY in left).toBe(false)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(false)

    // So when capacity returns it migrates normally, with no acceptance —
    // and the user's own query-settings key is still the authoritative one.
    restoreCapacityAndRerun(quotaStorage)

    expect(getSettingsMigrationError()).toBeNull()
    expect(parsedFrom(quotaStorage, QUERY_SETTINGS_STORAGE_KEY).state.querySettings.mode).toBe(
      'local'
    )
  })

  test('a PRE-EXISTING history key is never reclaimed (rule 4 still wins)', () => {
    const quotaStorage = new ByteQuotaStorage(2950, false)
    const userHistory = JSON.stringify({
      state: { history: [{ id: 'u1', role: 'user', content: 'the user own record' }] },
      version: 1
    })
    quotaStorage.setItem(WEBUI_RETRIEVAL_HISTORY_KEY, userHistory)
    quotaStorage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyWithBigHistory())

    runSettingsStorageSplitMigration(quotaStorage)

    // A key this run did NOT create survives untouched, whatever the quota did.
    expect(quotaStorage.removed).toEqual([])
    expect(quotaStorage.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(userHistory)
  })
})

describe('a shed envelope stamped POST-SPLIT still requires acceptance', () => {
  const shedAtPhaseTwo = () => {
    // Seed write + the three phase-1 target writes, then the budget collapses
    // to a window that fits neither the phase-2 payload nor the bytes just
    // removed — but does fit a scalar rung, which carries phase 2's OWN
    // version (22) along with the loss marker.
    const quotaStorage = new ByteQuotaStorage(100000, true, 4, 1400)
    quotaStorage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, {
        theme: 'dark',
        language: 'zh',
        apiKey: 'k'.repeat(400),
        querySettings: { mode: 'hybrid', top_k: 7 },
        retrievalHistory: [{ id: 'h1', role: 'user', content: 'x'.repeat(900) }]
      })
    )
    quotaStorage.budget = 3000
    runSettingsStorageSplitMigration(quotaStorage)
    return quotaStorage
  }

  test('the reload REFUSES instead of migrating the surviving fields', () => {
    // Regression (P1). The marker check used to sit BELOW the version guard,
    // and a rung shed at phase 2 keeps that payload's version — 22, already
    // past the cap. So the reload answered "post-split, nothing to migrate"
    // and returned clean, never reaching the marker: the partial settings
    // hydrated with no error and no acceptance, and the API key that did not
    // fit was gone silently. Exactly the loss the marker exists to report.
    const quotaStorage = shedAtPhaseTwo()

    // The run that did the shedding reports it…
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)
    const shed = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    expect(shed.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT) // past the cap
    expect(shed.lightragSettingsLostToQuota).toBe(true)
    expect(shed.state.apiKey).toBeUndefined() // the field that did not fit
    expect(shed.state.theme).toBe('dark') // what the rung saved

    // …and so does the reload, once there is room again.
    quotaStorage.budget = 100000
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(quotaStorage)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)
    // Refusing touched nothing.
    expect(parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)).toEqual(shed)
  })

  test('acceptance keeps the surviving fields and stops the refusal', () => {
    const quotaStorage = shedAtPhaseTwo()
    quotaStorage.budget = 100000

    acceptSettingsDataLoss(quotaStorage)

    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(quotaStorage)
    expect(getSettingsMigrationError()).toBeNull()
    const accepted = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    expect('lightragSettingsLostToQuota' in accepted).toBe(false)
    expect(accepted.state.theme).toBe('dark')
    expect(accepted.state.language).toBe('zh')
  })

  test('a field the split copies NOWHERE is a loss wherever it is shed', () => {
    // Regression (P1). The object rung's premise — "already written to their
    // own keys by phase 1" — holds only for the two fields the split
    // actually copies. `userPromptHistory` is not one of them: nothing
    // migrates it, the legacy chain writes it into every normalized
    // envelope, and it lives in this envelope exactly the way theme,
    // language and apiKey do. Shedding it used to be silent.
    //
    // Here every target already exists, so phase 1 writes nothing and the
    // quota can only bite on phase 2's cleanup — which means the shed rung
    // inherits that payload's v22 stamp. The refusal on reload therefore
    // depends on the marker being read BEFORE the version gate as well;
    // these two fixes hold this case together.
    const quotaStorage = new ByteQuotaStorage(100000, true)
    quotaStorage.setItem(
      QUERY_SETTINGS_STORAGE_KEY,
      JSON.stringify({ state: { querySettings: { mode: 'hybrid' } }, version: 1 })
    )
    quotaStorage.setItem(
      WORKSPACE_RETRIEVAL_HISTORY_KEY,
      JSON.stringify({ state: { history: [] }, version: 1 })
    )
    quotaStorage.setItem(
      WEBUI_RETRIEVAL_HISTORY_KEY,
      JSON.stringify({ state: { history: [] }, version: 1 })
    )
    quotaStorage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      legacyEnvelope(21, {
        theme: 'dark',
        language: 'zh',
        apiKey: 'secret',
        querySettings: { mode: 'hybrid', top_k: 7 },
        userPromptHistory: [
          'how does the graph index work?',
          'summarise the release notes'
        ],
        retrievalHistory: []
      })
    )
    // Mid-window (384..432): room for the scalar-only rung, not for the
    // payload that still carries the prompt history.
    quotaStorage.budget = 408

    runSettingsStorageSplitMigration(quotaStorage)

    const shed = parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)
    // The migrated field's own key is right there, so it is NOT what made
    // this a loss — the prompt history is.
    expect(quotaStorage.getItem(QUERY_SETTINGS_STORAGE_KEY)).not.toBeNull()
    expect(shed.state.userPromptHistory).toBeUndefined()
    expect(shed.state.theme).toBe('dark')
    expect(shed.version).toBe(SETTINGS_STORAGE_VERSION_AFTER_SPLIT)
    expect(shed[SETTINGS_LOST_MARKER_KEY]).toBe(true)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)

    // …and the reload refuses rather than completing over it.
    quotaStorage.budget = 100000
    resetSettingsMigrationErrorForTests()
    runSettingsStorageSplitMigration(quotaStorage)
    expect(isSettingsDataLostError(getSettingsMigrationError())).toBe(true)
    expect(parsedFrom(quotaStorage, LEGACY_SETTINGS_STORAGE_KEY)).toEqual(shed)
  })

  test('a STRICTLY future envelope is rule 3\'s, marker or not', () => {
    // The bound on reading the marker early. Refusing is not where rule 3
    // ends: the refusal routes the user to `acceptSettingsDataLoss`, which
    // WRITES — on a newer client's envelope it would rewrite those bytes to
    // strip a field this build does not own, and it would do so INSTEAD of
    // the future-envelope handling such an envelope is supposed to get.
    // A marker above this build's own version is the newer client's to
    // report, in its own build.
    storage.setItem(
      LEGACY_SETTINGS_STORAGE_KEY,
      JSON.stringify({
        state: { theme: 'dark' },
        version: SETTINGS_STORAGE_VERSION_AFTER_SPLIT + 1,
        [SETTINGS_LOST_MARKER_KEY]: true
      })
    )
    const before = storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)

    runSettingsStorageSplitMigration(storage)

    // Rule 3, unchanged: nothing read into a decision, nothing written…
    expect(getSettingsMigrationError()).toBeNull()
    expect(storage.getItem(LEGACY_SETTINGS_STORAGE_KEY)).toBe(before)
    expect(storage.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    // …and the envelope still reaches the guard that actually owns it.
    expect(hasFutureSettingsEnvelope(storage)).toBe(true)
  })
})

describe('query-settings store version stability', () => {
  // disable_user_prompt_prefix was added to defaultQuerySettings without a
  // version bump, deliberately. QUERY_SETTINGS_STORE_VERSION does double duty:
  // it is the persist migrate version AND the stamp the split writes
  // (splitSettingsStorage.ts:453). Bumping it makes split-derived envelopes
  // skip any `version < 2` migrate branch, so a backfill would have to be
  // duplicated inside writeTargetsAndClean.
  //
  // No backfill is needed anyway: an absent key hydrates as undefined, the
  // serializer omits it, and the server default applies the prefix — the
  // correct behaviour for an upgrading install.
  test('stays at 1', () => {
    expect(QUERY_SETTINGS_STORE_VERSION).toBe(1)
  })

  test('the split stamps query settings v1 and backfills no new key', () => {
    storage.setItem(LEGACY_SETTINGS_STORAGE_KEY, legacyEnvelope(21, v21State()))

    runSettingsStorageSplitMigration(storage)

    const envelope = parsedFrom(storage, QUERY_SETTINGS_STORAGE_KEY)
    expect(envelope.version).toBe(QUERY_SETTINGS_STORE_VERSION)
    // Carried over, not defaulted: the split copies the legacy settings and
    // leaves fields it has never heard of to the store's own defaults.
    expect(envelope.state.querySettings.mode).toBe('hybrid')
    expect('disable_user_prompt_prefix' in envelope.state.querySettings).toBe(false)
  })
})

const installLocalStorage = (stub: Storage): (() => void) => {
  const previousDescriptor = Object.getOwnPropertyDescriptor(globalThis, 'localStorage')
  Object.defineProperty(globalThis, 'localStorage', {
    value: stub,
    configurable: true
  })
  return () => {
    if (previousDescriptor) {
      Object.defineProperty(globalThis, 'localStorage', previousDescriptor)
    } else {
      delete (globalThis as Record<string, unknown>).localStorage
    }
  }
}

describe('split store disjoint writers', () => {
  // Storage-split isolation: the three split keys have disjoint writers, so a
  // write through one store must never touch the other keys — the whole-
  // envelope write-back race of the legacy single key is gone by construction.
  //
  // Isolation rule: the two history stores are FRESH instances built via the
  // exported factory while this block's stub is installed, while the (possibly
  // cached) query-settings singleton is read through its OWN
  // `persist.getOptions().storage`. Swapping `localStorage` in `beforeEach`/
  // `afterEach` avoids sibling-describe `beforeAll` hoisting in Bun.
  const stub = new FakeStorage()
  let restoreLocalStorage: (() => void) | undefined
  let webuiHistory: RetrievalHistoryStore | undefined
  let workspaceHistory: RetrievalHistoryStore | undefined
  let querySettings: typeof import('@/stores/querySettings').useQuerySettingsStore | undefined
  let savedQuerySettings:
    | ReturnType<typeof import('@/stores/querySettings').useQuerySettingsStore.getState>['querySettings']
    | undefined

  const readQuerySettingsEnvelope = async () => {
    const storeStorage = querySettings!.persist.getOptions().storage!
    return await storeStorage.getItem(QUERY_SETTINGS_STORAGE_KEY)
  }

  beforeEach(async () => {
    restoreLocalStorage = installLocalStorage(stub)
    if (!webuiHistory || !workspaceHistory || !querySettings) {
      const { createRetrievalHistoryStore } = await import('@/stores/retrievalHistory')
      webuiHistory = createRetrievalHistoryStore(WEBUI_RETRIEVAL_HISTORY_KEY)
      workspaceHistory = createRetrievalHistoryStore(WORKSPACE_RETRIEVAL_HISTORY_KEY)
      querySettings = (await import('@/stores/querySettings')).useQuerySettingsStore
      savedQuerySettings = { ...querySettings.getState().querySettings }
    }
  })

  afterEach(() => {
    restoreLocalStorage?.()
    restoreLocalStorage = undefined
  })

  afterAll(() => {
    if (querySettings && savedQuerySettings) {
      querySettings.setState({ querySettings: savedQuerySettings })
    }
  })

  test('workspace history writes touch ONLY the workspace history key', async () => {
    const settingsBefore = JSON.stringify(await readQuerySettingsEnvelope())
    const webuiBefore = stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)

    workspaceHistory!
      .getState()
      .setHistory([{ id: 'w1', role: 'user', content: 'workspace question' }])

    // Its own key (in this block's stub) got the write…
    expect(parsedFrom(stub, WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history).toHaveLength(1)
    // …and neither sibling key moved: not in this stub, and not wherever the
    // query-settings singleton actually persists.
    expect(stub.getItem(QUERY_SETTINGS_STORAGE_KEY)).toBeNull()
    expect(stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(webuiBefore)
    expect(JSON.stringify(await readQuerySettingsEnvelope())).toBe(settingsBefore)
  })

  test('the two histories are invisible to each other', () => {
    webuiHistory!
      .getState()
      .setHistory([{ id: 'a1', role: 'user', content: 'admin debug' }])

    expect(workspaceHistory!.getState().history.map((m) => m.id)).toEqual(['w1'])
    expect(webuiHistory!.getState().history.map((m) => m.id)).toEqual(['a1'])
    expect(parsedFrom(stub, WEBUI_RETRIEVAL_HISTORY_KEY).state.history[0].id).toBe('a1')
    expect(parsedFrom(stub, WORKSPACE_RETRIEVAL_HISTORY_KEY).state.history[0].id).toBe('w1')

    // Clearing one side leaves the other side's history untouched.
    workspaceHistory!.getState().clearHistory()
    expect(workspaceHistory!.getState().history).toEqual([])
    expect(webuiHistory!.getState().history.map((m) => m.id)).toEqual(['a1'])
    expect(parsedFrom(stub, WEBUI_RETRIEVAL_HISTORY_KEY).state.history[0].id).toBe('a1')
  })

  test('query settings writes touch ONLY the query-settings key', async () => {
    const webuiBefore = stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)
    const workspaceBefore = stub.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)

    querySettings!.getState().updateQuerySettings({ top_k: 99 })

    // Read back through the singleton's OWN storage binding.
    const envelope = (await readQuerySettingsEnvelope()) as {
      state: { querySettings: { top_k: number } }
    }
    expect(envelope.state.querySettings.top_k).toBe(99)
    // The history keys in this block's stub did not move.
    expect(stub.getItem(WEBUI_RETRIEVAL_HISTORY_KEY)).toBe(webuiBefore)
    expect(stub.getItem(WORKSPACE_RETRIEVAL_HISTORY_KEY)).toBe(workspaceBefore)
  })

  test('history_turns stays pinned to 0 through updates', () => {
    querySettings!.getState().updateQuerySettings({ history_turns: 7 } as never)
    expect(querySettings!.getState().querySettings.history_turns).toBe(0)
  })
})

describe('bounded history persistence under quota pressure', () => {
  // Bounded-capacity persistence for the retrieval histories (product
  // decision: the chat history is a non-critical test record). When a
  // persisted write exceeds the browser quota the write is ABANDONED — the
  // stored envelope resets to an empty history so the key stays bounded, the
  // in-memory conversation survives for the session, and a one-time toast
  // warns the user — instead of zustand's default of silently failing on
  // every subsequent write forever.
  const TEST_KEY = 'lightrag::quota-test-history'
  const quotaStub = new FakeStorage()
  quotaStub.failValueLongerThan = 500
  quotaStub.failureName = 'QuotaExceededError'

  let restoreLocalStorage: (() => void) | undefined
  let historyStore: RetrievalHistoryStore | undefined

  beforeEach(async () => {
    restoreLocalStorage = installLocalStorage(quotaStub)
    if (!historyStore) {
      const { createRetrievalHistoryStore } = await import('@/stores/retrievalHistory')
      historyStore = createRetrievalHistoryStore(TEST_KEY)
    }
  })

  afterEach(() => {
    restoreLocalStorage?.()
    restoreLocalStorage = undefined
  })

  test('an over-quota write abandons persistence but keeps the live conversation', () => {
    const bigHistory = [{ id: 'b1', role: 'user' as const, content: 'x'.repeat(2000) }]

    historyStore!.getState().setHistory(bigHistory)

    // The live session still shows the conversation…
    expect(historyStore!.getState().history).toHaveLength(1)
    // …while the persisted envelope was reset to an EMPTY history (bounded),
    // not left holding a stale value and not thrown out of setHistory.
    const persisted = parsedFrom(quotaStub, TEST_KEY)
    expect(persisted.version).toBe(RETRIEVAL_HISTORY_STORE_VERSION)
    expect(persisted.state.history).toEqual([])
  })

  test('small writes persist normally again afterwards', () => {
    const smallHistory = [{ id: 's1', role: 'user' as const, content: 'short' }]

    historyStore!.getState().setHistory(smallHistory)

    expect(parsedFrom(quotaStub, TEST_KEY).state.history).toHaveLength(1)
  })
})
