

import {promises as fsPromises} from 'fs';
import { exit } from 'process';
import puppeteer from 'puppeteer';

// Put exit(1)?
async function loadConfig() {
  try {
    let configFile = await fsPromises.open('scrape_config.example.json');
    let config = await configFile.readFile('utf-8').then(JSON.parse);

    let cookies = null;
    if (cookies_file in config) {
      cookies = JSON.parse(readFileSync(`cookies/${config.cookies_file}`, 'utf8'));
    }

    return {config, cookies};
  }
  catch (err) {
    console.log('Error: ', err);
    return {config: null, cookies: null};
  }
}


const {config, cookies} = await loadConfig();

if (!config) {
  console.log('No config file found.');
  exit(1);
}

// What items to select
const selector = '#gallery-gallery > figure > p:nth-child(1) > a'
// What to do with the selected items
const selectorFn = links => links.map(a => [a.href, a.innerText]);


// TODO: Grab titles as well as links!!
async function visitor(link, textData) {
    try {
      await page.goto(link[0]);
      const text = await page.$eval('.submission-description', desc => desc.innerText)
      textData[link[1]] = text;
    }
    catch (err) {
      console.log("Error: ", err)
    }
}

(async () => {
  const browser = await puppeteer.launch();
  const page = await browser.newPage();
  if (cookies) { await page.setCookie(...cookies); }

  // Don't load images.
  await page.setViewport({ width: 1920, height: 1080 });
  await page.setRequestInterception(true);
  page.on('request', (req) => { if (req.resourceType() === 'image') req.abort(); else req.continue(); });


  // Start browsing!
  await page.goto(config.target_url);
  const links = await page.$$eval(selector, selectorFn);

  const textData = {};
  for (const i in links) {
    console.log(`Visiting link ${i}/${links.length}`);
    visitor(links[i], textData);
  }

  await browser.close();

  try {
    await fsPromises.writeFile(
      `data/scrape/${config.output}`,
      JSON.stringify(textData, null, '  ')
    );
  } catch (err) {
    console.log('The file could not be written.', err);
  }

  console.log('Done.');
})();
