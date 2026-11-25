import { describe, expect, test } from 'bun:test'

import { hasDocumentWarning, hasKgRecoveryWarnings } from './documentRecoveryWarnings'
import { computeStatusCountsFingerprint } from './documentStatusCounts'
import { getStatusRequestFilters } from './documentStatusFilters'

describe('document recovery warning detection', () => {
  test('detects a processed document with structured recovery warnings', () => {
    expect(
      hasDocumentWarning({
        error_msg: '',
        metadata: {
          kg_recovery_warnings: [
            { code: 'degraded_custom_chunk_rollback', operation_id: 'op-1' }
          ]
        }
      })
    ).toBe(true)
  })

  test('keeps error messages as warnings', () => {
    expect(hasDocumentWarning({ error_msg: 'failed', metadata: {} })).toBe(true)
  })

  test('ignores absent, empty, and malformed recovery warnings', () => {
    expect(hasKgRecoveryWarnings(undefined)).toBe(false)
    expect(hasKgRecoveryWarnings({ kg_recovery_warnings: [] })).toBe(false)
    expect(hasKgRecoveryWarnings({ kg_recovery_warnings: ['not-structured'] })).toBe(false)
  })
})

describe('computeStatusCountsFingerprint', () => {
  test('sees a document moving between the two bucket-less statuses', () => {
    // The property the removed per-page bucket structure could not express.
    // `pending` and `preprocessed` have no filter bucket, so a pending document
    // becoming preprocessed changed none of that structure's keys and the
    // health check never fired. Every other count is held constant here, so
    // only sensitivity to those two statuses can make this pass.
    const before = computeStatusCountsFingerprint({
      all: 5,
      processed: 4,
      pending: 1,
      preprocessed: 0
    })
    const after = computeStatusCountsFingerprint({
      all: 5,
      processed: 4,
      pending: 0,
      preprocessed: 1
    })

    expect(after).not.toBe(before)
  })

  test('is stable for an unchanged map, regardless of key order and key case', () => {
    const reference = computeStatusCountsFingerprint({ all: 3, pending: 1, processed: 2 })

    expect(computeStatusCountsFingerprint({ processed: 2, all: 3, pending: 1 })).toBe(reference)
    expect(computeStatusCountsFingerprint({ PROCESSED: 2, all: 3, PENDING: 1 })).toBe(reference)
  })

  test('distinguishes a status appearing from it being absent', () => {
    expect(computeStatusCountsFingerprint({ all: 1, failed: 1 })).not.toBe(
      computeStatusCountsFingerprint({ all: 1 })
    )
  })
})

describe('documentStatusFilters', () => {
  test.each([
    ['deleting', 'deleting'],
    ['completed', 'processed'],
    ['parse', 'parsing'],
    ['analyze', 'analyzing'],
    ['process', 'processing'],
    ['failed', 'failed'],
    ['all', null]
  ] as const)('builds exact request filter for %s tab', (tab, expectedStatus) => {
    expect(getStatusRequestFilters(tab)).toEqual({
      status_filter: expectedStatus,
      status_filters: null
    })
  })
})
