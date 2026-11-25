/// <reference types="bun" />
import { describe, expect, test } from 'bun:test'
import { readFileSync } from 'fs'
import { join } from 'path'

import { assertDomAvailable, restoreDomGlobals, withoutDomGlobals } from './domGlobals'
import { childPath, relativePath, sourceFiles } from './sourceScan'

describe('assertDomAvailable', () => {
  test('says nothing when the preload installed a DOM', () => {
    expect(() => assertDomAvailable()).not.toThrow()
  })

  test('names the cause and the fix when there is no DOM', () => {
    withoutDomGlobals(() => {
      // The whole point of the guard is that the message is actionable: the
      // failure it replaces (`document[isPrepared]`, thrown from inside
      // user-event) names neither the working directory nor the preload.
      expect(() => assertDomAvailable()).toThrow(/lightrag_webui/)
      expect(() => assertDomAvailable()).toThrow(/bunfig\.toml/)
    })
  })
})

describe('withoutDomGlobals', () => {
  test('removes both globals by default and puts them back', () => {
    withoutDomGlobals(() => {
      expect(typeof globalThis.window).toBe('undefined')
      expect(typeof globalThis.document).toBe('undefined')
    })

    expect(typeof globalThis.window).not.toBe('undefined')
    expect(typeof globalThis.document).not.toBe('undefined')
  })

  test('removes only the named globals', () => {
    withoutDomGlobals(() => {
      expect(typeof globalThis.document).toBe('undefined')
      // `window` survives, which is what lets a test exercise "resolved a
      // title from the injected config, but has nowhere to write it".
      expect(typeof globalThis.window).not.toBe('undefined')
    }, ['document'])

    expect(typeof globalThis.document).not.toBe('undefined')
  })

  test('restores the globals even when the body throws', () => {
    expect(() =>
      withoutDomGlobals(() => {
        throw new Error('boom')
      })
    ).toThrow('boom')

    expect(typeof globalThis.document).not.toBe('undefined')
  })

  test('returns whatever the body returned', () => {
    expect(withoutDomGlobals(() => 42)).toBe(42)
  })
})

describe('restoreDomGlobals', () => {
  test('reinstates a stubbed global', () => {
    const real = globalThis.window
    Object.defineProperty(globalThis, 'window', {
      value: { __LIGHTRAG_CONFIG__: { webuiTitle: 'stub' } },
      configurable: true
    })
    expect(globalThis.window).not.toBe(real)

    restoreDomGlobals()

    expect(globalThis.window).toBe(real)
  })
})

describe('source scanning helpers', () => {
  test('childPath appends with / and leaves the root untouched', () => {
    // Teeth against NORMALIZING the root, which is the tempting wrong fix: a
    // backslash is a legal filename character on Linux, so rewriting one there
    // points the scan at a directory that does not exist.
    expect(childPath('C:\\repo\\src', 'lib')).toBe('C:\\repo\\src/lib')
    expect(childPath(childPath('C:\\repo\\src', 'lib'), 'loginIdentity.ts')).toBe(
      'C:\\repo\\src/lib/loginIdentity.ts'
    )
  })

  test('the walk never delegates to the platform join', () => {
    // On Linux `path.join` and a `/` template are the SAME function, so NO
    // assertion on this module's output can tell them apart here — the bug is
    // reachable only on Windows. Measured: reverting `childPath` to
    // `join(dir, name)` left the output tests at 4 pass. What is checkable on
    // every platform is that the module never reaches for `path` at all, which
    // is a genuine source-level property (a forbidden import), not a stand-in
    // for behaviour a render could show.
    const source = readFileSync(join(import.meta.dir, 'sourceScan.ts'), 'utf8')

    const specifiers = [...source.matchAll(/from\s*['"]([^'"]+)['"]/g)].map((m) => m[1])
    expect(specifiers).toEqual(['fs'])
  })

  test('relativePath returns the / tail even under a native-separator root', () => {
    // The regression this module exists for, as a pure function so it runs
    // the same on every platform: a Windows root reaches the audits with
    // backslashes, and what they compare against a literal is the tail.
    expect(relativePath('C:\\repo\\src', 'C:\\repo\\src/components/retrieval/mermaidLoader.ts')).toBe(
      'components/retrieval/mermaidLoader.ts'
    )
  })

  test('scanned tails are / separated and exclusions on them fire', () => {
    const dir = join(import.meta.dir, '..', 'lib')
    const files = sourceFiles(dir, (path) => path.endsWith('.ts'))

    expect(files.length).toBeGreaterThan(0)
    // The TAIL, deliberately, not the absolute path. The root is the caller's
    // own `path.join` output and still carries native separators on Windows —
    // left alone on purpose, since a backslash is a legal filename character
    // on Linux. Asserting no backslash anywhere in the absolute path failed
    // every Windows run of this suite while proving nothing on Linux.
    const tails = files.map((file) => relativePath(dir, file))
    expect(tails.filter((tail) => tail.includes('\\'))).toHaveLength(0)

    // A suffix exclusion is exactly what broke on Windows, so pin that one
    // works here rather than only that the paths look right.
    const identity = files.filter((file) => file.endsWith('/loginIdentity.ts'))
    expect(identity).toHaveLength(1)
  })

  test('accept decides what is kept, and directories are descended', () => {
    const dir = join(import.meta.dir, '..')
    const onlyTsx = sourceFiles(dir, (path) => path.endsWith('.tsx'))

    expect(onlyTsx.filter((file) => !file.endsWith('.tsx'))).toHaveLength(0)
    // Nested, so a walk that failed to recurse would come back short.
    expect(onlyTsx.some((file) => file.includes('/components/'))).toBe(true)
  })
})
