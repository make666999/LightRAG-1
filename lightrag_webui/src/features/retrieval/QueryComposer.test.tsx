/// <reference types="bun" />
import { afterEach, beforeEach, describe, expect, test } from 'bun:test'
import { cleanup, fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'

import { renderWithProviders } from '@/test/render'
import QueryComposer from './QueryComposer'

/**
 * The composer owns the validation notice: `onSend` returns a message to show
 * or null to accept. What matters is what the user is left with afterwards —
 * whether the message is announced, whether it survives editing, and whether
 * the draft is still there to correct — so it is exercised, not read.
 */

const REJECTED = 'Only supports the following query modes: naive, local'
const NOTICE_ID = 'query-input-error'

/** Anything starting with `/` is rejected, like a bad mode prefix. */
const onSend = (input: string): string | null => {
  sent.push(input)
  return input.startsWith('/') ? REJECTED : null
}

let sent: string[]

const renderComposer = () =>
  renderWithProviders(
    <QueryComposer
      isLoading={false}
      stopDisabled={false}
      onSend={onSend}
      onStop={() => {}}
      onClear={() => {}}
    />
  )

const input = (): HTMLElement => within(screen.getByRole('search')).getByRole('textbox')
const inputEl = (): HTMLInputElement | HTMLTextAreaElement =>
  input() as HTMLInputElement | HTMLTextAreaElement
const notice = (): HTMLElement => screen.getByRole('alert')

beforeEach(() => {
  sent = []
})

afterEach(() => {
  cleanup()
})

describe('query composer multiline editing', () => {
  // Browsers run setTimeout(0) before the next animation frame; happy-dom runs
  // them the other way round. Without this stub the caret-restoring rAF in
  // QueryComposer fires before the handlers' setTimeout, so a test would only
  // pin the inputRef.current half of the fix. Scope the stub to this describe
  // block: bun runs every test file in one process, so a leaked global would
  // change the timing of unrelated tests.
  const realRaf = globalThis.requestAnimationFrame

  beforeEach(() => {
    globalThis.requestAnimationFrame = (cb: FrameRequestCallback) =>
      setTimeout(() => cb(performance.now()), 16) as unknown as number
  })

  afterEach(() => {
    globalThis.requestAnimationFrame = realRaf
  })

  // The stubbed frame is a 16ms timer. Let it run before asserting or typing,
  // otherwise it can move the caret back in the middle of the next keystroke.
  const settleFrame = () => new Promise((resolve) => setTimeout(resolve, 24))
  const moveTo = (pos: number) => `{Home}${'{ArrowRight}'.repeat(pos)}`

  const seedMultiline = async (
    user: ReturnType<typeof userEvent.setup>,
    firstLine: string,
    secondLine: string
  ): Promise<HTMLTextAreaElement> => {
    await user.type(input(), firstLine)
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    await settleFrame()
    await user.type(input(), secondLine)
    await settleFrame()
    return input() as HTMLTextAreaElement
  }

  type MultilineSwitchSpec = {
    initial: string
    settleBeforeEdit?: boolean
    caretAt?: number
    edit: { kind: 'shiftEnter' } | { kind: 'paste'; text: string }
    settleAfterEdit: boolean
    expectedSelection?: number
    followUp: string
    expectedValue: string
  }

  // Settled cases (A, B, F) let the stubbed frame run before follow-up typing;
  // same-frame cases (G, H, I) model a fast typist whose next keystrokes arrive
  // before rAF fires, so the swapped element must already own focus and caret.
  const switchCases: Array<[string, MultilineSwitchSpec]> = [
    [
      'A: Shift+Enter keeps typing after the inserted newline',
      {
        initial: 'alpha beta',
        caretAt: 5,
        edit: { kind: 'shiftEnter' },
        settleAfterEdit: true,
        expectedSelection: 6,
        followUp: 'NEW',
        expectedValue: 'alpha\nNEW beta'
      }
    ],
    [
      'B: pasting a newline keeps typing at the end of the pasted text',
      {
        initial: 'alpha beta',
        caretAt: 5,
        edit: { kind: 'paste', text: 'x\ny' },
        settleAfterEdit: true,
        expectedSelection: 8,
        followUp: 'Z',
        expectedValue: 'alphax\nyZ beta'
      }
    ],
    [
      'F: Shift+Enter at position 0 keeps typing after the newline',
      {
        initial: 'alpha',
        caretAt: 0,
        edit: { kind: 'shiftEnter' },
        settleAfterEdit: true,
        expectedSelection: 1,
        followUp: 'X',
        expectedValue: '\nXalpha'
      }
    ],
    [
      'G: typing right after Shift+Enter in the middle lands after the newline',
      {
        initial: 'alpha beta',
        settleBeforeEdit: true,
        caretAt: 5,
        edit: { kind: 'shiftEnter' },
        settleAfterEdit: false,
        followUp: 'NEW',
        expectedValue: 'alpha\nNEW beta'
      }
    ],
    [
      'H: typing right after Shift+Enter at the end lands after the newline',
      {
        initial: 'alpha',
        settleBeforeEdit: true,
        edit: { kind: 'shiftEnter' },
        settleAfterEdit: false,
        followUp: 'NEW',
        expectedValue: 'alpha\nNEW'
      }
    ],
    [
      'I: typing right after a multi-line paste lands after the pasted text',
      {
        initial: 'alpha beta',
        settleBeforeEdit: true,
        caretAt: 5,
        edit: { kind: 'paste', text: 'x\ny' },
        settleAfterEdit: false,
        followUp: 'Z',
        expectedValue: 'alphax\nyZ beta'
      }
    ]
  ]

  test.each(switchCases)('%s', async (_name, {
    initial,
    settleBeforeEdit,
    caretAt,
    edit,
    settleAfterEdit,
    expectedSelection,
    followUp,
    expectedValue
  }) => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), initial)
    if (settleBeforeEdit) await settleFrame()
    if (caretAt !== undefined) await user.keyboard(moveTo(caretAt))

    if (edit.kind === 'shiftEnter' && !settleAfterEdit) {
      await user.keyboard(`{Shift>}{Enter}{/Shift}${followUp}`)
    } else {
      if (edit.kind === 'shiftEnter') await user.keyboard('{Shift>}{Enter}{/Shift}')
      else await user.paste(edit.text)

      if (settleAfterEdit) {
        await settleFrame()
        expect(input().tagName).toBe('TEXTAREA')
        expect(inputEl().selectionStart).toBe(expectedSelection!)
      }
      await user.keyboard(followUp)
    }

    expect(inputEl().value).toBe(expectedValue)
    expect(sent).toEqual([])
  })

  test('C: forward Delete at position 0 keeps typing at the start', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), 'alpha')
    await settleFrame()
    await user.keyboard('{Home}{Delete}')
    await settleFrame()
    await user.keyboard('X')

    // A 0 caret is a real position: `selectionStart || length` would send the
    // next keystroke to the end of the draft instead.
    expect(inputEl().value).toBe('Xlpha')
    expect(sent).toEqual([])
  })

  test('D: Backspace removing the only newline keeps the caret on the new input', async () => {
    const user = userEvent.setup()
    renderComposer()

    const textarea = await seedMultiline(user, 'alpha', ' beta')
    expect(textarea.value).toBe('alpha\n beta')
    textarea.setSelectionRange(6, 6)
    await user.keyboard('{Backspace}')
    await settleFrame()

    expect(input().tagName).toBe('INPUT')
    expect(inputEl().selectionStart).toBe(5)
    await user.keyboard('X')

    expect(inputEl().value).toBe('alphaX beta')
    expect(sent).toEqual([])
  })

  test('J: pasting over an identical selection collapses the caret to its end', async () => {
    const user = userEvent.setup()
    renderComposer()

    const textarea = await seedMultiline(user, 'a', 'b')
    // The pasted text equals the selection, so the draft does not change and
    // React does not re-render: the caret must still move past the paste, and
    // nothing may be left over to move it again on a later keystroke.
    textarea.setSelectionRange(0, 3)
    await user.paste('a\nb')
    await settleFrame()
    await user.keyboard('Z')
    await settleFrame()
    await user.keyboard('Y')

    expect(inputEl().value).toBe('a\nbZY')
    expect(sent).toEqual([])
  })

  test('K: Shift+Enter and paste inside a textarea keep the caret without a switch', async () => {
    const user = userEvent.setup()
    renderComposer()

    const textarea = await seedMultiline(user, 'ab', 'cd')
    // Already a textarea: these edits re-render the same element, so the
    // caret restore must not depend on the element type changing.
    textarea.setSelectionRange(1, 1)
    await user.keyboard('{Shift>}{Enter}{/Shift}X')
    await user.paste('1\n2')
    await user.keyboard('Y')

    expect(input() === textarea).toBe(true)
    expect(textarea.value).toBe('a\nX1\n2Yb\ncd')
    expect(sent).toEqual([])
  })
})

describe('query composer input method composition', () => {
  for (const draft of ['中文问题', '第一行\n中文问题']) {
    for (const keyboardState of [
      { isComposing: true },
      { isComposing: false, keyCode: 229 }
    ]) {
      test(`candidate confirmation preserves ${JSON.stringify(draft)} with ${JSON.stringify(keyboardState)}`, () => {
        renderComposer()
        if (draft.includes('\n')) {
          fireEvent.keyDown(input(), { key: 'Enter', shiftKey: true })
          expect(input().tagName).toBe('TEXTAREA')
        }
        fireEvent.change(input(), { target: { value: draft } })
        fireEvent.keyDown(input(), { key: 'Enter', code: 'Enter', ...keyboardState })

        expect(sent).toEqual([])
        expect(inputEl().value).toBe(draft)

        fireEvent.keyDown(input(), { key: 'Enter', code: 'Enter', isComposing: false })
        expect(sent).toEqual([draft])
        expect(inputEl().value).toBe('')
      })
    }
  }
})

describe('query composer validation notice', () => {
  test('a rejected send announces an alert above the composer, keeps the draft and wires assistive attributes', async () => {
    const user = userEvent.setup()
    renderComposer()

    expect(input().getAttribute('aria-invalid')).toBe(null)
    expect(input().getAttribute('aria-describedby')).toBe(null)

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')

    // `role="alert"` is what makes a screen reader speak this without the user
    // going looking for it, and the draft stays so it can be corrected.
    expect(notice().textContent).toContain(REJECTED)
    expect(inputEl().value).toBe('/nope hello')
    expect(input().getAttribute('aria-invalid')).toBe('true')
    expect(input().getAttribute('aria-describedby')).toBe(NOTICE_ID)
    expect(notice().getAttribute('id')).toBe(NOTICE_ID)

    // Below the composer it would sit off-screen on a short viewport, under
    // the mobile keyboard, or behind the send button.
    const classes = (notice().getAttribute('class') ?? '').split(/\s+/)
    expect(classes).toContain('absolute')
    expect(classes).toContain('bottom-full')
    expect(classes).not.toContain('top-full')
  })

  test('the notice survives editing and carries wiring across the multi-line swap', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')
    expect(input().tagName).toBe('INPUT')

    // It names what to change; wiping it on the first keystroke would leave
    // the user correcting blind.
    await user.type(input(), ' more')
    expect(screen.queryAllByRole('alert')).toHaveLength(1)
    expect(notice().textContent).toContain(REJECTED)

    // Shift+Enter switches the single-line Input for a Textarea. The swap is
    // exactly where the wiring gets lost: the second element is a different
    // node and inherits none of the first one's attributes.
    await user.keyboard('{Shift>}{Enter}{/Shift}')
    expect(input().tagName).toBe('TEXTAREA')
    expect(input().getAttribute('aria-invalid')).toBe('true')
    expect(input().getAttribute('aria-describedby')).toBe(NOTICE_ID)
  })

  test('an accepted send clears the notice and the draft', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')
    expect(screen.queryAllByRole('alert')).toHaveLength(1)

    await user.clear(input())
    await user.type(input(), 'a real question')
    await user.keyboard('{Enter}')

    expect(sent).toEqual(['/nope hello', 'a real question'])
    expect(screen.queryAllByRole('alert')).toHaveLength(0)
    expect(inputEl().value).toBe('')
    expect(input().getAttribute('aria-invalid')).toBe(null)
  })

  test('the notice can be dismissed without editing', async () => {
    const user = userEvent.setup()
    renderComposer()

    await user.type(input(), '/nope hello')
    await user.keyboard('{Enter}')

    await user.click(within(notice()).getByRole('button', { name: 'Dismiss error' }))

    expect(screen.queryAllByRole('alert')).toHaveLength(0)
    // Dismissing is not sending: the draft is still the user's to fix.
    expect(inputEl().value).toBe('/nope hello')
    expect(sent).toEqual(['/nope hello'])
  })
})
