import { readFileSync, writeFile } from 'fs';
import { chromium } from 'playwright'; // 1. Switch to Playwright's chromium

const cookies = JSON.parse(readFileSync('cookies/www.furaffinity.net.cookies.json', 'utf8'));
const target = 'https://www.furaffinity.net/gallery/honovy/';
const out = 'scrape/claweddays.json';

(async () => {

  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 }, // Block images using route handler
    bypassCSP: true // Often needed for proper request interception
  });

  // Block images
  await context.route(/.*\.(png|jpg|jpeg|webp|svg|gif)$/, route => route.abort());

  await context.addCookies(cookies);

  console.log('Loading FA');
  const page = await context.newPage();
  await page.goto(target);

  const links = await page.$$eval(
    '#gallery-gallery > figure > p:nth-child(1) > a',
    links => links.map(a => [a.href, a.innerText])
  );
  console.log(links);

  const textData = {};
  for(const i in links) {
    console.log(`Browsing to link ${i}/${links.length}`);
    try {
      await page.goto(links[i][0]);
      const text = await page.$eval(
        '.submission-description',
        desc => desc.innerText
      );
      textData[links[i][1]] = text;
    }
    catch (err) {
      console.log("Error: ", err);
    }
  }

  await browser.close();

  writeFile(out, JSON.stringify(textData, null, "  "), (err) => {
    if (err) throw err;
    console.log('The file has been saved!');
  });

  console.log('Done.');
})();
