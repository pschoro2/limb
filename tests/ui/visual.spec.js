import { expect, test } from "@playwright/test";

test("renders nonblank 3D scene and stereo panels", async ({ page }) => {
  await page.goto("/");
  const canvas = page.locator("#scene");
  await expect(canvas).toBeVisible();
  await expect(page.locator("#left-eye")).toBeVisible();
  await expect(page.locator("#right-eye")).toBeVisible();
  await expect(page.locator("#status-action")).toContainText(/HOLD|RETRACT|APPROACH|PROBE|TOUCH|WITHDRAW|ORIENT/);

  await page.waitForTimeout(600);
  const sample = await canvas.evaluate((node) => {
    const context = node.getContext("webgl2") || node.getContext("webgl");
    const width = node.width;
    const height = node.height;
    const pixels = new Uint8Array(width * height * 4);
    context.readPixels(0, 0, width, height, context.RGBA, context.UNSIGNED_BYTE, pixels);
    let nonBackground = 0;
    for (let index = 0; index < pixels.length; index += 4 * 97) {
      const r = pixels[index];
      const g = pixels[index + 1];
      const b = pixels[index + 2];
      if (Math.abs(r - 21) + Math.abs(g - 27) + Math.abs(b - 33) > 14) nonBackground += 1;
    }
    return { width, height, nonBackground };
  });

  expect(sample.width).toBeGreaterThan(300);
  expect(sample.height).toBeGreaterThan(300);
  expect(sample.nonBackground).toBeGreaterThan(20);
});

test("dragging a pointy object into the sensors teaches pain and triggers withdrawal", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#scene")).toBeVisible();
  await dragObjectTowardLimb(page, "knife", -1.0);

  await expect(page.locator("#selected-label")).toContainText("knife");
  await expect(page.locator("#status-action")).toContainText("WITHDRAW_FAST");
  await expect(page.locator("#left-eye-meta")).not.toContainText("no object");
  await expect(page.locator("#sensor-state")).toContainText(/pain|closing/);

  const state = await page.evaluate(() => ({
    limb: window.limbSimDebug.limbPosition(),
    sensors: window.limbSimDebug.sensors(),
    memory: window.limbSimDebug.memoryEntries(),
  }));
  expect(state.limb.y).toBeCloseTo(-0.12, 1);
  expect(state.sensors.pain).toBeGreaterThan(0.55);
  expect(state.memory.some((entry) => entry.risk > 0.55)).toBe(true);

  await page.mouse.up();
});

test("dragging a spoon toward the limb triggers safe reach", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#scene")).toBeVisible();
  await dragObjectTowardLimb(page, "spoon", 0.0);

  await expect(page.locator("#selected-label")).toContainText("spoon");
  await expect(page.locator("#status-action")).toContainText(/APPROACH_SLOW|GENTLE_TOUCH/);

  const state = await page.evaluate(() => ({
    limb: window.limbSimDebug.limbPosition(),
    memory: window.limbSimDebug.memoryEntries(),
  }));
  expect(state.limb.y).toBeCloseTo(-0.12, 1);
  expect(state.memory.some((entry) => entry.label === "spoon" && entry.valence > 0)).toBe(true);

  await page.mouse.up();
});

async function dragObjectTowardLimb(page, key, z) {
  await page.waitForFunction((objectKey) => window.limbSimDebug?.objectScreenPosition(objectKey), key);
  const { start, end } = await page.evaluate(
    ({ objectKey, targetZ }) => ({
      start: window.limbSimDebug.objectScreenPosition(objectKey),
      end: window.limbSimDebug.worldScreenPosition(0.95, -0.27, targetZ),
    }),
    { objectKey: key, targetZ: z },
  );
  expect(start).not.toBeNull();
  expect(end).not.toBeNull();

  await page.mouse.move(start.x, start.y);
  await page.mouse.down();
  await page.mouse.move(end.x, end.y, { steps: 24 });
  await page.waitForTimeout(450);
}
