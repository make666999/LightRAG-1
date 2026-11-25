import { afterAll, afterEach, beforeAll, beforeEach, describe, expect, mock, test } from 'bun:test'
import { act, cleanup, fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'

import { renderWithProviders } from '@/test/render'
import { useAuthStore } from '@/stores/state'
import { resetCustomization, seedCustomization } from '@/test/customization'

const BUNDLE_LOGO = 'https://cdn.example.test/acme-logo.png'
const DOCUMENT_HEADING = 'Acme Terms of Service'
const DOCUMENT_BODY = 'You agree to the acceptable use policy.'
const LINK_TEXT = 'Terms and Privacy Policy'

/**
 * The page reconciles the session against `/auth-status` on mount, through
 * the axios-based API module. Import ORDER is what makes the stub effective:
 * `mock.module` only reaches importers that have not been evaluated yet, so
 * the page is imported dynamically AFTER it — a static `import LoginPage`
 * at the top of this file would bind the real module first and the request
 * would go out for real (`stores/state.test.ts` uses the same ordering).
 * The module is restored in afterAll so later test files get the real one.
 */
let realApiModule: Record<string, unknown>
let LoginPage: typeof import('./LoginPage').default

/**
 * What `/auth-status` reports for the next render. Read INSIDE the mock so a
 * single file-level stub can serve tests that need different responses; the
 * `webui_title` here is a live value from the server, which the page must
 * prefer over anything cached when the bundle names no title of its own.
 */
let authStatus: { auth_configured: boolean; webui_title?: string } = {
  auth_configured: true
}

beforeAll(async () => {
  realApiModule = { ...(await import('@/api/lightrag')) }
  mock.module('@/api/lightrag', () => ({
    ...realApiModule,
    getAuthStatus: mock(async () => authStatus)
  }))
  LoginPage = (await import('./LoginPage')).default
})

afterAll(() => {
  mock.module('@/api/lightrag', () => realApiModule)
})

beforeEach(() => {
  authStatus = { auth_configured: true }
  // These are pre-login pages; an earlier test file that activates a session
  // would otherwise leave them rendering nothing. The precondition belongs
  // to the test, not to whatever ran before it.
  act(() => {
    useAuthStore.setState({ isAuthenticated: false, isGuestMode: false })
  })
})

afterEach(() => {
  // Unmount FIRST. A file-local `afterEach` runs BEFORE the preload's
  // Testing Library `cleanup()`, so resetting the store here would update a
  // still-mounted `useCustomizedContent`, whose effect immediately calls
  // `load(locale)` and starts a real /ui/customization request during
  // teardown — one that can land during the NEXT test and overwrite its
  // seeded snapshot. `cleanup` is idempotent, so the preload's own call is
  // still fine.
  cleanup()
  resetCustomization()
})

/**
 * Render the login page and wait for the customization verdict to settle.
 *
 * The page deliberately renders NOTHING until then (workspace-entry PRD
 * §8.8: never flash the default content before knowing whether a bundle is
 * active), so every assertion here has to wait first.
 */
const renderLogin = async () => {
  const view = renderWithProviders(
    <MemoryRouter>
      <LoginPage />
    </MemoryRouter>
  )
  await waitFor(() => {
    if (!view.container.innerHTML) throw new Error('login page has not settled')
  }, {
    // A real (refused) /auth-status attempt can outlast the 1s default
    // when the whole suite runs in one process.
    timeout: 5000
  })
  return view
}

describe('login page branding', () => {
  test('shows the bundle logo with the bundle alt text', async () => {
    seedCustomization({ logo_url: BUNDLE_LOGO, logo_alt: 'Acme Corp' })
    const { container } = await renderLogin()

    const logo = await screen.findByRole('img', { name: 'Acme Corp' })
    expect(logo).toHaveAttribute('src', BUNDLE_LOGO)

    // And it is the ONLY logo. Finding the bundle image says nothing about
    // what sits beside it: LightRAG's built-in asset or the lightning glyph
    // rendered alongside would put product branding on a customer's login
    // page, which is exactly what customization exists to replace.
    expect(screen.getAllByRole('img')).toHaveLength(1)
    expect(container.querySelectorAll('.lucide-zap')).toHaveLength(0)
  })

  test('falls back to the title /auth-status just reported when the bundle names none', async () => {
    // The bundle is the preferred source, but an uncustomized deployment can
    // still set WEBUI_TITLE. That value arrives with the auth-status response
    // this page load made, so it is the FRESHEST title available and has to
    // reach the heading — the whole reason the page threads the auth-status
    // title back into `useCustomizedContent`.
    authStatus = { auth_configured: true, webui_title: 'Ops Knowledge Base' }
    seedCustomization({}, { customized: false })
    await renderLogin()

    expect(
      await screen.findByRole('heading', { name: 'Ops Knowledge Base' })
    ).toBeInTheDocument()
  })

  test('the bundle title still wins over the auth-status one', async () => {
    authStatus = { auth_configured: true, webui_title: 'Ops Knowledge Base' }
    seedCustomization({ title: 'Acme Knowledge Base' })
    await renderLogin()

    expect(
      await screen.findByRole('heading', { name: 'Acme Knowledge Base' })
    ).toBeInTheDocument()
    expect(screen.queryAllByRole('heading', { name: 'Ops Knowledge Base' })).toHaveLength(0)
  })

  test('a failed logo does not block a different logo from a later locale', async () => {
    seedCustomization({ logo_url: BUNDLE_LOGO, logo_alt: 'Acme Corp' })
    await renderLogin()

    // A bundle can name a URL this browser cannot fetch (expired CDN, blocked
    // host): the logo is dropped rather than left as a broken image.
    fireEvent.error(await screen.findByRole('img', { name: 'Acme Corp' }))
    await waitFor(() => {
      expect(screen.queryAllByRole('img', { name: 'Acme Corp' })).toHaveLength(0)
    })

    // The failure is latched against the URL that failed, not against
    // "logos": switching language loads a different bundle whose logo has
    // never been tried, and it must still render.
    act(() => {
      seedCustomization({ logo_url: 'https://cdn.example.test/acme-fr.png', logo_alt: 'Acme France' })
    })

    expect(await screen.findByRole('img', { name: 'Acme France' })).toHaveAttribute(
      'src',
      'https://cdn.example.test/acme-fr.png'
    )
  })
})

describe('login agreements dialog', () => {
  const seedConsentBundle = (consentDocuments: string | null = LINK_TEXT): void => {
    seedCustomization(
      { title: 'Acme KB' },
      {
        consent_required: true,
        consent_documents: consentDocuments,
        agreements: {
          format: 'markdown',
          content: `# ${DOCUMENT_HEADING}\n\n${DOCUMENT_BODY}\n`
        }
      }
    )
  }

  const openAgreements = async () => {
    const user = userEvent.setup()
    await user.click(await screen.findByRole('button', { name: LINK_TEXT }))
    return screen.findByRole('dialog')
  }

  test('the consent link opens the agreement document with accurate naming, single visible title and document typography', async () => {
    seedConsentBundle()
    await renderLogin()

    const dialog = await openAgreements()

    // The document itself is what the visitor came to read; it is rendered,
    // not summarised or linked away to.
    expect(within(dialog).getByText(DOCUMENT_BODY)).toBeInTheDocument()

    // The accessible name is the link text the bundle maintains beside the
    // file — not a title parsed out of the Markdown, which would drift from
    // the wording on the checkbox the moment either changed.
    expect(dialog).toHaveAccessibleName(LINK_TEXT)

    // The dialog's own title element exists ONLY to give the dialog its
    // accessible name, so it is screen-reader-only; the one title a sighted
    // visitor sees belongs to the document. A printed one would sit above
    // the document's own heading and disagree with it.
    const namingTitle = within(dialog).getByText(LINK_TEXT)
    expect(namingTitle.closest('.sr-only') !== null).toBe(true)

    const visibleHeadings = within(dialog)
      .getAllByRole('heading')
      .filter((heading) => heading.closest('.sr-only') === null)
    expect(visibleHeadings.map((heading) => heading.textContent)).toEqual([DOCUMENT_HEADING])

    // `compact` is the DEFAULT variant, and it tightens paragraph and heading
    // margins for short blurbs sitting inside another layout. A legal
    // document is the page content here and has to read like the Markdown
    // file it came from, so the wrapper must carry neither of those.
    const prose = dialog.querySelector('.prose')
    expect(prose === null).toBe(false)
    const classes = prose!.className
    expect(classes.includes('prose-p:my-2')).toBe(false)
    expect(classes.includes('prose-headings:my-3')).toBe(false)
  })

  test('falls back to the translated wording when the bundle names no documents', async () => {
    seedConsentBundle(null)
    await renderLogin()

    // 'login.consentDocuments' in locales/en.json. The gate never depends on
    // the bundle supplying wording — only on consent_required.
    const trigger = await screen.findByRole('button', { name: 'Privacy Policy Agreement' })
    expect(trigger).toBeInTheDocument()
  })
})
