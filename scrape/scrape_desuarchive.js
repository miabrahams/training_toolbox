import fs from 'fs';
import { chromium } from 'playwright';

const outfile = 'scrape/desuarchive_msdg_20240331.json';
const startTarget = 'https://desuarchive.org/trash/thread/62548034';

(async () => {
  console.log('Starting Playwright.');
  const browser = await chromium.launch();
  console.log('Opening tab.');
  const page = await browser.newPage();

  console.log('Disabling images.');
  await page.setViewportSize({ width: 1920, height: 1080 });
  await page.setUserAgent('Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/78.0.3904.108 Safari/537.36');

  // No image dl
  await page.route('**/*', route => {
    if (route.request().resourceType() === 'image') {
      route.abort();
    } else {
      route.continue();
    }
  });

  let textData;
  try {
    const fileContent = await fs.promises.readFile(outfile, 'utf8');
    textData = JSON.parse(fileContent);
  } catch (err) {
    console.log('No file found, starting fresh.');
    textData = {};
  }

  let currentUrl = startTarget;

  for (let pageCount = 0; pageCount < 20; pageCount++) {
    console.log(`Browsing to link ${pageCount}/20 - ${currentUrl}`);
    await page.goto(currentUrl);
    console.log('Page loaded.');

    const page_links = await page.$$eval('a.post_file_filename', (elements) =>
      elements.map(el => ({ link: el.href, filename: el.innerText }))
    );
    console.log(`found ${page_links.length} links.`);

    const page_id = currentUrl.split('/').slice(-1)[0];
    textData[page_id] = page_links;

    try {
      currentUrl = await page.$eval('::-p-text(Previous thread) a', el => el.href);
    } catch (error) {
      console.warn("Couldn't find 'Previous thread' link: stopping.");
      break;
    }
  }

  await browser.close();

  console.log(`Next URL would be: ${currentUrl}`);

  try {
    await fs.promises.writeFile(outfile, JSON.stringify(textData, null, "  "));
    console.log('The file has been saved!');
  } catch (err) {
    console.log('The file could not be written.', err);
  }

  console.log('Done.');
})();
