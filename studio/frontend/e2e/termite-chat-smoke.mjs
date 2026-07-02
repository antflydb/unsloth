import { chromium } from 'playwright';

const baseUrl = process.env.UNSLOTH_STUDIO_URL ?? 'http://127.0.0.1:8888';
const password = process.env.UNSLOTH_STUDIO_PASSWORD ?? 'Neoteny7';
const modelPath = process.env.UNSLOTH_TERMITE_MODEL ?? 'unsloth/Llama-3.2-1B-Instruct-GGUF';
const ggufVariant = process.env.UNSLOTH_TERMITE_VARIANT ?? 'Q4_K_M';
const prompt = process.env.UNSLOTH_TERMITE_PROMPT ?? 'Reply with one short sentence.';
const screenshotPath = process.env.UNSLOTH_TERMITE_SCREENSHOT ?? '/tmp/termite-chat-smoke.png';
const assistantSelector = '[data-role="assistant"]';
const generatingOnlyTexts = new Set(['Generating...']);

function fail(message) {
  throw new Error(message);
}

const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1728, height: 1117 } });

let chatRequestSeen = false;
let chatStatus = null;
let chatFailure = null;
page.on('response', async (response) => {
  if (!response.url().includes('/v1/chat/completions')) return;
  chatRequestSeen = true;
  chatStatus = response.status();
  if (chatStatus >= 400) {
    try {
      chatFailure = await response.text();
    } catch {
      chatFailure = '<failed to read body>';
    }
  }
});
page.on('pageerror', (err) => {
  console.error('PAGEERROR', err);
});

async function clickNewChatIfPresent(page) {
  const button = page.getByRole('button', { name: 'New Chat' }).first();
  if (await button.isVisible().catch(() => false)) {
    await button.click();
  }
}

try {
  await page.goto(`${baseUrl}/login`, { waitUntil: 'networkidle' });
  await page.getByLabel('Password').fill(password).catch(async () => {
    await page.locator('input[type="password"]').fill(password);
  });
  await page.getByRole('button', { name: /login/i }).click();
  await page.waitForURL('**/chat', { timeout: 20000 });

  const setup = await page.evaluate(async ({ modelPath, ggufVariant }) => {
    const token = localStorage.getItem('unsloth_auth_token');
    if (!token) return { ok: false, error: 'missing unsloth_auth_token' };

    async function call(path, body) {
      const resp = await fetch(path, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`,
        },
        body: JSON.stringify(body),
      });
      const text = await resp.text();
      return { ok: resp.ok, status: resp.status, text };
    }

    const backend = await call('/api/inference/backend', { backend: 'termite-zig' });
    if (!backend.ok) return { ok: false, stage: 'backend', ...backend };

    const load = await call('/api/inference/load', {
      model_path: modelPath,
      gguf_variant: ggufVariant,
      max_seq_length: 4096,
    });
    if (!load.ok) return { ok: false, stage: 'load', ...load };

    return { ok: true, backend, load };
  }, { modelPath, ggufVariant });

  if (!setup.ok) {
    fail(`setup failed at ${setup.stage ?? 'unknown'}: status=${setup.status ?? '?'} body=${setup.text ?? setup.error}`);
  }

  await page.reload({ waitUntil: 'networkidle' });
  await clickNewChatIfPresent(page);

  const backendLabel = await page.getByRole('combobox', { name: 'Inference engine' }).innerText();
  if (!backendLabel.includes('Antfly')) {
    fail(`expected Antfly picker label, got: ${backendLabel}`);
  }

  const assistantCountBefore = await page.locator(assistantSelector).count();
  await page.getByLabel('Message input').fill(prompt);
  await page.getByRole('button', { name: 'Send message' }).click();

  await page.waitForFunction(
    ({ selector, before }) => document.querySelectorAll(selector).length > before,
    { selector: assistantSelector, before: assistantCountBefore },
    { timeout: 30000 },
  );

  await page.waitForFunction(
    ({ selector, badTexts }) => {
      const nodes = [...document.querySelectorAll(selector)];
      const last = nodes.at(-1);
      if (!last) return false;
      const text = (last.textContent || '').trim();
      return text.length > 0 && !badTexts.includes(text);
    },
    { selector: assistantSelector, badTexts: [...generatingOnlyTexts] },
    { timeout: 30000 },
  );
  await page.waitForTimeout(2000);

  if (!chatRequestSeen) {
    fail('no /v1/chat/completions request observed');
  }
  if (chatStatus !== 200) {
    fail(`chat returned status ${chatStatus}: ${chatFailure ?? '<no body>'}`);
  }

  const assistantText = (await page.locator(assistantSelector).last().innerText()).trim();
  if (!assistantText || generatingOnlyTexts.has(assistantText)) {
    fail(`chat UI did not render assistant text; assistant=${JSON.stringify(assistantText)}`);
  }

  await page.screenshot({ path: screenshotPath, fullPage: true });
  console.log(JSON.stringify({ ok: true, backendLabel, modelPath, ggufVariant, assistantText, screenshotPath }));
} finally {
  await browser.close();
}
