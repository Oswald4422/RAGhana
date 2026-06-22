import asyncio
from playwright.async_api import async_playwright

BASE_URL = "https://raghana-ai.onrender.com"
OUT_DIR = "resources"

PROMPTS = [
    "NPP votes in Ashanti Region 2020",
    "Who won the 2016 presidential election in Ghana?",
    "What is Ghana's debt-to-GDP target in the 2025 budget?",
    "Total NDC votes across all regions in 2012",
    "Revenue mobilisation strategy in the 2025 budget",
    "CPP votes in Volta Region 2016",
    "What was the education expenditure in the 2025 national budget?",
]


async def wait_for_response(page):
    """Wait until the loading indicator disappears and a response appears."""
    await page.wait_for_selector(".response-text, .answer, [class*='response'], [class*='answer'], p",
                                 timeout=90000)
    # Extra wait for content to settle
    await page.wait_for_timeout(2000)


async def type_and_submit(page, prompt):
    textarea = await page.query_selector("textarea, input[type='text'], input:not([type])")
    if textarea:
        await textarea.click()
        await textarea.fill("")
        await textarea.fill(prompt)
        await page.wait_for_timeout(500)
        await textarea.press("Enter")
    else:
        # Try a submit button
        button = await page.query_selector("button[type='submit'], button")
        if button:
            await button.click()


async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport={"width": 1400, "height": 900})
        page = await context.new_page()

        # 1. Homepage screenshot
        print("Taking homepage screenshot...")
        await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
        await page.wait_for_timeout(3000)
        await page.screenshot(path=f"{OUT_DIR}/01_homepage.png", full_page=True)
        print("  Saved 01_homepage.png")

        # 2-8. Test prompts
        for i, prompt in enumerate(PROMPTS, start=2):
            print(f"Running prompt {i}: {prompt[:50]}...")
            await page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
            await page.wait_for_timeout(2000)

            await type_and_submit(page, prompt)

            # Wait up to 90s for a response to appear
            try:
                await page.wait_for_timeout(8000)  # initial wait for response to start
                # Wait for loading spinner to disappear if present
                try:
                    await page.wait_for_selector(
                        "[class*='loading'], [class*='spinner'], [class*='thinking']",
                        state="hidden",
                        timeout=90000
                    )
                except Exception:
                    pass
                await page.wait_for_timeout(3000)
            except Exception as e:
                print(f"  Warning: {e}")

            fname = f"{OUT_DIR}/{i:02d}_prompt_{i-1}.png"
            await page.screenshot(path=fname, full_page=True)
            print(f"  Saved {fname}")

        await browser.close()
        print("\nAll 8 screenshots saved to resources/")


asyncio.run(main())
